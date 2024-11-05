//reciever.message
//a-z lowercase => global variables by default


(
var number;
number = 3.cubed;
number;
)

~number = 3.cubed;
~number;

(
var foo;
foo = 3;
foo = foo.cubed;
foo = foo+100;
foo = foo/2;
foo;
)


Server.local.boot;
Server.local.quit;

s.boot;

//{
//functions:
//filled with one or more unit generators (UGens) which are calculations with signals (for example oscilators)
//}




x = {SinOsc.ar}; //function
y = x.play;//this will create a synth and send it to the audio synthesisi program. syhths are individual entities that exists on the audio server
y.free; //free synth from the audio server. Only Synths can be set free. Not functions


//UGenerators: calculations with signals
//ar -> audio rate. Outputs values at sample rate (44100Hz). Use for hearing the output of a UGen
//kr -> kontrol rate. Outputs 1 value for every 64 audio samples. Use for UGens that shape or controlls other UGens
//ir -> Don't output values consinuously. Just output one value when the Synth is instantiated


(
z = {
	arg freq =440, amp =1;
	var sig;
	sig = SinOsc.ar(freq)*amp;
}.play;
)

z.set(\freq, 220);
z.set(\amp, 0.5);
z.free



(
z = {
	var freq ,sig;
	freq = LFNoise0.kr(8).exprange(200,1000);
	sig = SinOsc.ar(freq);
}.play;
)

z.set(\freq, 220);
z.set(\amp, 0.5);
z.free

//The proper way of creating a sound is creating a Synth Def(~recipie for a sound) and a Synths is the execution of that recipie





o = OSCFunc({ |msg| msg.postln }, '/tr', s.addr);


{    var    freq;
    freq = LFNoise1.kr(2, 600, 800);
        // Impulse is needed to trigger the /tr message to be sent
    SendTrig.kr(Impulse.kr(4), 0, freq);
    SinOsc.ar(freq, 0, 0.3) ! 2
}.play;
[ /tr, 1000, 0, 1340.8098144531 ]
[ /tr, 1000, 0, 1153.9201660156 ]
[ /tr, 1000, 0, 966.35247802734 ]
[ /tr, 1000, 0, 629.31628417969 ]

o.free;  // when done, you need to clean up the OSCFunc



s.quit;