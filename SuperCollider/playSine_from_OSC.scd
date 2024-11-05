
s.boot;

// Create a control bus for the frequency
~freqBus = Bus.control(s, 1);  // A control bus with 1 channel

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq, {
    var freq = In.kr(~freqBus);  // Read frequency from control bus
    var sound = SinOsc.ar(freq) * 0.6;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Initialize the synth and connect it to the frequency bus
~sineSynth = Synth(\sineFreq);
~freqBus.set(440);  // Set an initial frequency, e.g., 440 Hz

// OSCFunc to update the frequency on the control bus
(
OSCFunc({
    |msg|
    var frequency = msg[1];  // Get the frequency from the message
    ~freqBus.set(frequency);  // Update the frequency on the bus
    ("Frequency updated to: " ++ frequency).postln;  // Optional: print the frequency
}, '/freq').fix;
)

s.quit;