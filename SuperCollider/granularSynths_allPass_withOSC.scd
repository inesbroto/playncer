
s.boot()
// 2 synthdefs - the 1st to make grains and the 2nd to delay them
​
// the synthdef that makes the grains is on the left channel
// the synthdef that delays the grains is on the right channel

(
~freqBus = Bus.control(s, 1);  // A control bus with 1 channel
~freqDevBus = Bus.control(s, 1);
~grainFreqBus = Bus.control(s, 1);
~ampBus = Bus.control(s, 1);
~delayAllPassBus = Bus.control(s, 1);
~decayAllPassBus = Bus.control(s, 1);

)
// OSCFunc to update the frequency on the control bus
(
OSCFunc({
    |msg|
    var frequency = msg[1];  // Get the frequency from the message
    ~freqBus.set(frequency);  // Update the frequency on the bus
    ("Frequency updated to: " ++ frequency).postln;  // Optional: print the frequency
}, '/freq').fix;
OSCFunc({
    |msg|
    var freqDev = msg[1];
    ~freqDevBus.set(freqDev);
    ("FreqDev updated to: " ++ freqDev).postln;
}, '/freqDev').fix;
OSCFunc({
    |msg|
    var grainFreq = msg[1];
    ~grainFreqBus.set(grainFreq);
    ("Frequency updated to: " ++ grainFreq).postln;
}, '/grainFreq').fix;
OSCFunc({
    |msg|
    var amp = msg[1];
    ~ampBus.set(amp);
    ("Amplitude updated to: " ++ amp).postln;
}, '/amp').fix;
OSCFunc({
    |msg|
    var delay = msg[1];
    ~delayAllPassBus.set(delay);
    ("Delay updated to: " ++ delay).postln;
}, '/delayAllPass').fix;
OSCFunc({
    |msg|
    var decay = msg[1];
    ~decayAllPassBus.set(decay);
    ("Decay updated to: " ++ decay).postln;
}, '/decayAllPass').fix;
)



(
SynthDef("Grains", {
	var gate, envelope,centerFreq = In.kr(~freqBus),grainFreq = In.kr(~grainFreqBus),freqDev = In.kr(~freqDevBus), amp = In.kr(~ampBus);
    gate = Impulse.kr(grainFreq); //control-rate impulse signal
	envelope = Env.sine(0.2); //creates a sine-shaped amplitude envelope with a duration of 0.2 seconds
	Out.ar(
		0,
		SinOsc.ar(
			LFNoise0.kr(4, freqDev, centerFreq),
			0,
			EnvGen.kr(envelope, gate, amp)
		)
	)
}).add;
​
SynthDef("aDelay", { //arg delay = 0.25;
	var delay = In.kr(~ampBus);
    Out.ar(
        1,
        DelayN.ar(
            In.ar(0, 1),
            delay,
            delay
        )
    )
}).add;


SynthDef("allpass", { //arg gain = 0.2;
	var delay = In.kr(~delayAllPassBus), decay = In.kr(~decayAllPassBus);
    Out.ar(
        1,
        AllpassN.ar(
			In.ar(0, 1),
			delay,
			delay,
			decay
		)
    )
}).add;

)
​
// make 2 groups, the 1st for sources and the 2nd for effects
(
~source = Group.head(s);
~effects= Group.tail(s);

)
​
(
Synth.head(~source, "Grains");
Synth.head(~effects, "allpass");
)

~ampBus.set(0.1);
~freqBus.set(340);
~freqDevBus.set(50);
~grainFreqBus.set(10);
~delayAllPassBus.set(0.01)
~decayAllPassBus.set(10)








~source.free;
s.quit()

~source.set(\grainFreq,2);

~source.reboot;

~source.resume;


x.free;
~source.free;

s.quit()