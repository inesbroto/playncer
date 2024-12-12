


s.boot;

// Create a control bus for the frequency
~freqBus1 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus2 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus3 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus4 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus5 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus6 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus7 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus8 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus9 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus10 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus11 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus12 = Bus.control(s, 1);  // A control bus with 1 channel
~freqBus13 = Bus.control(s, 1);  // A control bus with 1 channel



// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq1, {
    var freq = In.kr(~freqBus1);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq2, {
    var freq = In.kr(~freqBus2);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq3, {
    var freq = In.kr(~freqBus3);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq4, {
    var freq = In.kr(~freqBus4);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq5, {
    var freq = In.kr(~freqBus5);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq6, {
    var freq = In.kr(~freqBus6);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq7, {
    var freq = In.kr(~freqBus7);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq8, {
    var freq = In.kr(~freqBus8);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq9, {
    var freq = In.kr(~freqBus9);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq10, {
    var freq = In.kr(~freqBus10);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq11, {
    var freq = In.kr(~freqBus11);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq12, {
    var freq = In.kr(~freqBus12);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Define a SynthDef that reads frequency from the control bus
SynthDef(\sineFreq13, {
    var freq = In.kr(~freqBus13);  // Read frequency from control bus
    var sound = LFTri.ar(freq) * 0.2;  // Sine wave with the control frequency
    Out.ar(0, sound!2);  // Output to speakers in stereo
}).add;

// Initialize the synth and connect it to the frequency bus
~sineSynth = Synth(\sineFreq1);
~sineSynth = Synth(\sineFreq2);
~sineSynth = Synth(\sineFreq3);
~sineSynth = Synth(\sineFreq4);
~sineSynth = Synth(\sineFreq5);
~sineSynth = Synth(\sineFreq6);
~sineSynth = Synth(\sineFreq7);
~sineSynth = Synth(\sineFreq8);
~sineSynth = Synth(\sineFreq9);
~sineSynth = Synth(\sineFreq10);
~sineSynth = Synth(\sineFreq11);
~sineSynth = Synth(\sineFreq12);
~sineSynth = Synth(\sineFreq13);
//~freqBus.set(440);  // Set an initial frequency, e.g., 440 Hz

// OSCFunc to update the frequency on the control bus
(
OSCFunc({
    |msg|
    var frequency1 = msg[1];  // Get the frequency from the message
    ~freqBus1.set(frequency1);  // Update the frequency on the bus
    ("Frequency 1 updated to: " ++ frequency1 ++ msg).postln;  // Optional: print the frequency
}, '/freq1').fix;
)
(
OSCFunc({
    |msg|
    var frequency2 = msg[1];  // Get the frequency from the message
    ~freqBus2.set(frequency2);  // Update the frequency on the bus
    ("Frequency 2 updated to: " ++ frequency2).postln;  // Optional: print the frequency
}, '/freq2').fix;
)

(
OSCFunc({
    |msg|
    var frequency3 = msg[1];  // Get the frequency from the message
    ~freqBus3.set(frequency3);  // Update the frequency on the bus
    ("Frequency 3 updated to: " ++ frequency3).postln;  // Optional: print the frequency
}, '/freq3').fix;
)
(
OSCFunc({
    |msg|
    var frequency4 = msg[1];  // Get the frequency from the message
    ~freqBus4.set(frequency4);  // Update the frequency on the bus
    ("Frequency 4 updated to: " ++ frequency4).postln;  // Optional: print the frequency
}, '/freq4').fix;
)
(
OSCFunc({
    |msg|
    var frequency5 = msg[1];  // Get the frequency from the message
    ~freqBus5.set(frequency5);  // Update the frequency on the bus
    ("Frequency 5 updated to: " ++ frequency5).postln;  // Optional: print the frequency
}, '/freq5').fix;
)
(
OSCFunc({
    |msg|
    var frequency6 = msg[1];  // Get the frequency from the message
    ~freqBus6.set(frequency6);  // Update the frequency on the bus
    ("Frequency 6 updated to: " ++ frequency6).postln;  // Optional: print the frequency
}, '/freq6').fix;
)
(
OSCFunc({
    |msg|
    var frequency7 = msg[1];  // Get the frequency from the message
    ~freqBus7.set(frequency7);  // Update the frequency on the bus
    ("Frequency7 updated to: " ++ frequency7).postln;  // Optional: print the frequency
}, '/freq7').fix;
)
(
OSCFunc({
    |msg|
    var frequency8 = msg[1];  // Get the frequency from the message
    ~freqBus8.set(frequency8);  // Update the frequency on the bus
    ("Frequency8 updated to: " ++ frequency8).postln;  // Optional: print the frequency
}, '/freq8').fix;
)
(
OSCFunc({
    |msg|
    var frequency9 = msg[1];  // Get the frequency from the message
    ~freqBus9.set(frequency9);  // Update the frequency on the bus
    ("Frequency9 updated to: " ++ frequency9).postln;  // Optional: print the frequency
}, '/freq9').fix;
)
(
OSCFunc({
    |msg|
	var frequency10 = msg[1];  // Get the frequency from the message
    ~freqBus10.set(frequency10);  // Update the frequency on the bus
    ("Frequency 10 updated to: " ++ frequency10).postln;  // Optional: print the frequency
}, '/freq10').fix;
)
(
OSCFunc({
    |msg|
    var frequency11 = msg[1];  // Get the frequency from the message
    ~freqBus11.set(frequency11);  // Update the frequency on the bus
    ("Frequency 11 updated to: " ++ frequency11).postln;  // Optional: print the frequency
}, '/freq11').fix;
)
(
OSCFunc({
    |msg|
    var frequency12 = msg[1];  // Get the frequency from the message
    ~freqBus12.set(frequency12);  // Update the frequency on the bus
    ("Frequency 12 updated to: " ++ frequency12).postln;  // Optional: print the frequency
}, '/freq12').fix;
)
(
OSCFunc({
    |msg|
    var frequency13 = msg[1];  // Get the frequency from the message
    ~freqBus13.set(frequency13);  // Update the frequency on the bus
    ("Frequency 13 updated to: " ++ frequency13).postln;  // Optional: print the frequency
}, '/freq13').fix;
)


s.quit;















































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
	// Parse the message
    // Extract the ID from the OSC address, e.g., "/freq1" -> id = 1
	var osc_id = msg[1];//.asString;//.split("_")[0];//.asString.replace("/freq", "")//.asInteger;
	var frequency = msg[1];//.asString;//.split("_")[1]; // Frequency from the message

    ~freqBus.set(frequency);  // Update the frequency on the bus
	("Frequency updated to: " ++ frequency ++ " for osc num" ++ osc_id).postln;  // Optional: print the frequency
	(msg).postln;  // Optional: print the frequency

}, '/freq').fix;
)

s.quit;










// Dictionary to store synths
~oscillators = Dictionary.new;

// Define a SynthDef for sine oscillators
SynthDef(\sineOsc, { |out = 0, freq = 440, amp = 0.1|
    var sig = SinOsc.ar(freq) * amp;
    Out.ar(out, sig);
}).add;

// OSCdef to handle incoming frequency messages
OSCdef(\freqHandler, { |msg, time, addr, port|
    var id, freq;

    // Parse the message
    // Extract the ID from the OSC address, e.g., "/freq1" -> id = 1
    id = msg[0].asString.replace("/freq", "").asInteger;
    freq = msg[1]; // Frequency from the message

    // Check if the oscillator already exists
    if (~oscillators[id].notNil,
        // Update frequency of existing oscillator
        ~oscillators[id].set(\freq, freq),
        // Create a new oscillator if it doesn't exist
        ~oscillators[id] = Synth(\sineOsc, [\freq, freq]);
	)
}, '/freq');







// Example: Summing all oscillators into a single output
// You can route this through a bus or mixer.
OSCdef(\stop, { |msg|
    ~oscillators.do { |osc| osc.free };
    ~oscillators.clear;
    "All oscillators stopped.".postln;
}, '/stop');


s.quit;