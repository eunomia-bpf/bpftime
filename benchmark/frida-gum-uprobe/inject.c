#include <frida-core.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>

int
main (int argc, char * argv[])
{
  int result = 0;
  const char * path = "./agent.so";
  FridaInjector * injector;
  int pid;
  GError * error;
  guint id;

  frida_init ();

  if (argc != 2)
    goto bad_usage;

  pid = atoi (argv[1]);
  if (pid <= 0)
    goto bad_usage;

  injector = frida_injector_new ();
  printf("pid %d\n", pid);

  error = NULL;
  id = frida_injector_inject_library_file_sync (injector, pid, path, "example_agent_main", "example data", NULL, &error);
  if (error != NULL)
  {
    g_printerr ("%s\n", error->message);
    g_clear_error (&error);

    result = 1;
  }
  printf("id %d\n", id);
  frida_injector_close_sync (injector, NULL, NULL);
  g_object_unref (injector);

  frida_deinit ();

  return result;

bad_usage:
  {
    g_printerr ("Usage: %s <pid>\n", argv[0]);
    frida_deinit ();
    return 1;
  }
}
