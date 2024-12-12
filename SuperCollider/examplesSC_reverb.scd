

s.boot()
// 2 synthdefs - the 1st to make grains and the 2nd to delay them
​
// the synthdef that makes the grains is on the left channel
// the synthdef that delays the grains is on the right channel

(
SynthDef("someGrains", { arg centerFreq = 777, freqDev = 200, grainFreq = 2;
    var gate;
    gate = Impulse.kr(grainFreq);
    Out.ar(
        0,
        SinOsc.ar(
            LFNoise0.kr(4, freqDev, centerFreq),
            0,
            EnvGen.kr(Env.sine(0.1), gate, 0.1)
        )
    )
}).add;
​
SynthDef("aDelay", { arg delay = 0.25;
    Out.ar(
        1,
        DelayN.ar(
            In.ar(0, 1),
            delay,
            delay
        )
    )
}).add;
)
​
////////////////////////////////////////////////
// test the grains ... and then turn them off
// ... they're all on the left channel ... good!
Synth("someGrains");
////////////////////////////////////////////////
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


x.free;
~source.free;

s.quit()



s.boot()
/////////////////////////////////////////////////////////////////////////////////////////
// TURN ON THE INTERNAL SERVER!!
// first a comb filter and then an allpass with (with the same parameters) - compare them
/////////////////////////////////////////////////////////////////////////////////////////

// comb example
(
{
    CombN.ar(
        SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
        0.3,
        0.25,
        6
    )
}.scope;


// allpass example - not much difference from the comb example
(
{
    AllpassN.ar(
        SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
        0.3,
        0.25,
        6
    )
}.scope;
)
)


/////////////////////////////////////////////////////////////////////////////////////////
//
// first a comb example and then an allpass
// both examples have the same parameters
// the 2 examples have relatively short delay times ... 0.1 seconds
//
/////////////////////////////////////////////////////////////////////////////////////////
s.boot()

// comb
(
{
    CombN.ar(
        SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
        0.1,
        0.025,
        6
    )
}.scope;
)

// allpass ... what's the difference between this example and the comb filter?
(
{
    AllpassN.ar(
        SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
        0.1,
        0.025,
        6
    )
}.scope
)


{
var s, z, y;
    // 10 voices of a random sine percussion sound :
s = Mix.ar(Array.fill(10, { Resonz.ar(Dust.ar(0.2, 50), 200 + 3000.0.rand, 0.003)}) );
    // reverb predelay time :
z = DelayN.ar(s, 0.048);
    // 7 length modulated comb delays in parallel :
y = Mix.ar(Array.fill(7,{ CombL.ar(z, 0.1, LFNoise1.kr(0.1.rand, 0.04, 0.05), 15) }));
    // two parallel chains of 4 allpass delays (8 total) :
4.do({ y = AllpassN.ar(y, 0.050, [0.050.rand, 0.050.rand], 1) });
    // add original sound to reverb and play it :
s+(0.2*y)
}.scope

s.free;
s.quit()





////////////////////////////////////

s.boot()

(
SynthDef("filteredDust", {
    Out.ar(
        2,
        Mix.arFill(10, { Resonz.ar(Dust.ar(2, 50), Rand(200, 3200), 0.003) })
    )
}).add;

SynthDef("preDelay", {
    ReplaceOut.ar(
        4,
        DelayN.ar(In.ar(2, 1), 0.048, 0.048)
    )
}).add;

SynthDef("combs", {
    ReplaceOut.ar(
        6,
        Mix.arFill(7, { CombL.ar(In.ar(4, 1), 0.1, LFNoise1.kr(Rand(0, 0.1), 0.04, 0.05), 15) })
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

SynthDef("theMixer", { arg gain = 1;
    ReplaceOut.ar(
        0,
        Mix.ar([In.ar(2, 1), In.ar(8, 2)]) * gain
    )
}).add;

)

s.quit()
(
~source = Group.tail(s);
~proc1 = Group.tail(s);
~proc2 = Group.tail(s);
~proc3 = Group.tail(s);
~final = Group.tail(s);
)

// the nodes, below, are assigned to the groups, as ordered above,
(
Synth.head(~final, "theMixer");
Synth.head(~proc3, "allpass");
Synth.head(~proc2, "combs");
Synth.head(~proc1, "preDelay");
Synth.head(~source, "filteredDust");
)


~source.free;


(

~dustRoutine = Routine({
    loop {
        Synth("filteredDust", target: ~source);
        0.5.wait; // Wait before re-triggering
    }
})
)

~dustRoutine.play;

(
s.queryAllNodes;
)

~source.free;


