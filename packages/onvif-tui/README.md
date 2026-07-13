<h3>onvif-tui</h3>

This is a Terminal User Interface application that demonstrates libonvif abilities and can be used to evaluate and control camera settings. The application is launched using the command line arguments:

```
-u username for camera authentication
-p passwword for camera authentication
-m camera ip address for manual camera discovery
-i host local ip address for binding discovery broadcast 
```

Please visit the [github repository](https://github.com/sr99622/libonvif) for more detailed information.


<h2>Development Instructions</h2>

<b>For Windows, use Powershell</b>

<h2>Using uv</h2>


Install the Textual dev tools into the uv environment:

```
uv add --dev textual-dev
uv sync
```

Use two terminals.

From the first terminal, start the Textual console. The -X flags limit messages, the output should be only the print statements from the program:

```
uv run textual console -x SYSTEM -x EVENT -x DEBUG -x INFO -x LOGGING -x WORKER
```

Then from the second terminal, set the environment variable for TEXTUAL, this only needs to be done once per terminal session. The syntax will depend on the operating system:

* <b>Windows</b>
```
$env:TEXTUAL = "debug,devtools"
```

* <b>Mac and Linux</b>
```
export TEXTUAL="debug,devtools"
```

Now in the second terminal you can run the program, and the STDOUT from the program will be shown on the first terminal running the server.

```
uv run onvif-tui
```

<h2>Using standard python</h2>

Use two terminals

in the first terminal, start the console server

```
textual console -x SYSTEM -x EVENT -x DEBUG -x INFO -x LOGGING -x WORKER
```

In the second terminal, start the program

```
textual run --dev my_app.py
```

&nbsp;
