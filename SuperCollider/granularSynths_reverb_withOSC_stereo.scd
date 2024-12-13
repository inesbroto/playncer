
s.boot()
// 2 synthdefs - the 1st to make grains and the 2nd to delay them
​
// the synthdef that makes the grains is on the left channel
// the synthdef that delays the grains is on the right channel

(
~freqBus = Bus.control(s, 1);  // A control bus with 1 channel
~freqDevBus = Bus.control(s, 1);  // A control bus with 1 channel
~grainFreqBus = Bus.control(s, 1);  // A control bus with 1 channel
~delayBus = Bus.control(s, 1);  // A control bus with 1 channel

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
    var freqDev = msg[1];  // Get the frequency from the message
    ~freqDevBus.set(freqDev);  // Update the frequency on the bus
    ("FreqDev updated to: " ++ freqDev).postln;  // Optional: print the frequency
}, '/freqDev').fix;
OSCFunc({
    |msg|
    var grainFreq = msg[1];  // Get the frequency from the message
    ~grainFreqBus.set(grainFreq);  // Update the frequency on the bus
    ("Frequency updated to: " ++ grainFreq).postln;  // Optional: print the frequency
}, '/grainFreq').fix;
OSCFunc({
    |msg|
    var delay_ms = msg[1];  // Get the frequency from the message
    ~delayBus.set(delay_ms);  // Update the frequency on the bus
    ("delay updated to: " ++ delay_ms).postln;  // Optional: print the frequency
}, '/delay_ms').fix;
)



(
SynthDef("someGrains", { arg amp = 1;
    var gate, signal,centerFreq = In.kr(~freqBus),grainFreq = In.kr(~grainFreqBus),freqDev = In.kr(~freqDevBus);  // Read frequency from control bus
    gate = Impulse.kr(grainFreq);

	signal = SinOsc.ar(
			LFNoise0.kr(4, freqDev, centerFreq),
			0,
			EnvGen.kr(Env.sine(0.1), gate, 0.1) * amp
		);

	Out.ar(0,[signal,signal]);
}).add;
​
SynthDef("aDelay", { arg delay = In.kr(~delayBus);
    var input, delayedSignal;
	input = In.ar(0, 2);
	delayedSignal = DelayN.ar(
            input,
            delay,
            delay
        );
	Out.ar(
        0, input + delayedSignal
    )
}).add;

SynthDef("allpass", { arg gain = 0.2;
    var source;
    source = In.ar(6, 1);
    4.do({ source = AllpassN.ar(source, 0.050, [Rand(0, 0.05), Rand(0, 0.05)], 1) });
    ReplaceOut.ar(
        8,
        source * gain
    )
}).add;
)
​

// make 2 groups, the 1st for sources and the 2nd for effects
(
~source = Group.head(s);
~effects = Group.tail(s);
)
​
// place grains into the delay ... source is on the left and delayed source is on the right
(
Synth.head(~source, "someGrains");
Synth.head(~effects, "aDelay");
)

~delayBus.set(500000)
~freqBus.set(400);  // Set an initial frequency, e.g., 440 Hz
~freqDevBus.set(500); 
~grainFreqBus.set(100); 




~source.free;
s.quit()

~source.set(\grainFreq,2);
~source.reboot;
~source.resume;


x.free;
~source.free;

s.quit()