
s.boot;

// Create a control bus for the frequency
~freqBus = Bus.control(s, 1);  // A control bus with 1 channel
~amplBus = Bus.control(a, 1);  // A control bus with 1 channel

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq, {
    var freq = In.kr(~freqBus);  // Read frequency from control bus
	var ampl = In.kr(~amplBus);  // Read amplitude from control bus

    var sound = SinOsc.ar(freq) * ampl;  // Sine wave with the control frequency and amplitude
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Initialize the synth and connect it to the frequency bus
~sineSynth = Synth(\sineFreq);
~freqBus.set(440);  // Set an initial frequency, e.g., 440 Hz
~amplBus.set(0.6);  // Set an initial frequency, e.g., 440 Hz

// OSCFunc to update the frequency on the control bus
(
OSCFunc({
    |msg|
    var frequency = msg[1];  // Get the frequency from the message
    ~freqBus.set(frequency);  // Update the frequency on the bus
    ("Frequency updated to: " ++ frequency).postln;  // Optional: print the frequency
}, '/freq').fix;
)


(
OSCFunc({
    |msg|
    var amplitude = msg[1];  // Get the amplitude from the message
    ~amplBus.set(amplitude);  // Update the amplitude on the bus
    ("Amplitude updated to: " ++ amplitude).postln;  // Optional: print the amplitude
}, '/amp').fix;
)

s.quit;