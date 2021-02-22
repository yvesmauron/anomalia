Posenet-tfjs
===================
POSe net estimation using TensorFlow JavaScript.
Draws a skeleton over the current person in front a the webcam.

posenet-tfjs
├── posenet_githubdemo                 - download demo version from github
│   ├── docu                           - online documentation
│   └── README.md
├── posenet_sandbox
│   ├── local_includes_npm             - downloaded files from codesandbox
│   └── Sandbox_Tfjsposenet.html       - forward to Sandbox : codesandbox
├── posenet_twilio
│   └── assets
│       ├── webcam_tfjs.html           - Version tfjs - source code
│       ├── webcam_tfjsnode.html       - Version tfjs - binaries
│       └── webcam_tfjsnodegpu.html    - Version tfjs - cuda support
└── Readme_Posenet.txt


posenet_githubdemo
---------------------
The local installation of the libs is quite hard to do correctly.
Requirements : local apache server and javascript libraries


posenet_sandbox
--------------------
This runs well, but you need to use GoogleChrome.
It does not work in a different internet browser. 
Server: Codesandbox.co

Cloud installation:
sandbox with posenet-tfjs
---------------------
Source https://codesandbox.io/s/tfjs-posenet-toz3w
sandbox development

Local Installation:
Requirements : local apache server and javascript libraries


posenet_twilio
---------------
Source : https://github.com/elizabethsiegle/twilioVideoWebChat9Mins/tree/master
This a projekt for the posenet webcam.
The posenet libs are taken from cloud.
Requirements : local apache server

Pro: also run under Firefox
Con: You need GoogleChrome, else it is damn slow


