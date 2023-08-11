# android-inject-custom

Example showing how to use Frida for standalone injection of a custom
payload. The payload is a .so that uses Gum, Frida's low-level instrumentation
library, to hook `open()` and print the arguments on `stderr` every time it's
called. The payload could be any shared library as long as it exports a function
with the name that you specify when calling `inject_library_file_sync()`.

In our example we named it `example_agent_main`. This function will also be
passed a string of data, which you can use for application-specific purposes.

Note that only the build system is Android-specific, so this example is
easily portable to all other OSes supported by Frida.

# Prerequisites

- Android NDK r21
- Rooted Android device

# Preparing the build environment

Point `$ANDROID_NDK_ROOT` to your NDK path.

# Running

```sh
$ make
```

This will build the injector, the payload, and an example program you
can inject the payload into to easily observe the results.

Next copy the `bin/` directory somewhere on your Android device, and in one
terminal adb shell into your device and launch the `victim` binary:

```sh
$ ./victim
Victim running with PID 1303
```

Then in another terminal change directory to where the `inject` binary
is and run it:

```sh
$ sudo ./inject 1303
```

You should now see a message printed by the `victim` process every time
`open()` is called.

## performance

```
elapsed_time 0.010000ms
res 0
elapsed_time 0.011000ms
res 0
elapsed_time 0.010000ms
res 0
example_agent_main()
elapsed_time 1.649000ms
res 0
elapsed_time 1.639000ms
res 0
elapsed_time 38.333000ms
```

```
$ bin/victim 
Victim running with PID 13190
open 0x7f1f1370c9f0, close 0x7f1f1370d440
my_test_func 0x55faddb892cf
elapsed_time 0.020000ms
res 0
elapsed_time 0.009000ms
res 0
elapsed_time 34.517000ms
res 0
elapsed_time 40.670000ms
res 0
elapsed_time 37.135000ms
res 0
```