

int _bpf_helper_ext_0006(char* arg0);

int bpf_main(void *ctx)
{
	_bpf_helper_ext_0006("target_func called.\n");
	return 0;
}

