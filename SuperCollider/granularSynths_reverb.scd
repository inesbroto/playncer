
s.boot()
// 2 synthdefs - the 1st to make grains and the 2nd to delay them
​
// the synthdef that makes the grains is on the left channel
// the synthdef that delays the grains is on the right channel

SynthDef("someGrains", { arg centerFreq = 777, freqDev = 200, grainFreq = 10;
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


(
SynthDef("someGrains", { arg centerFreq = 777, freqDev = 200, grainFreq = 2, amp = 1;
    var gate;
    gate = Impulse.kr(grainFreq);
    Out.ar(
        0,
        SinOsc.ar(
            LFNoise0.kr(4, freqDev, centerFreq),
            0,
            EnvGen.kr(Env.sine(0.1), gate, 0.1) * amp
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
////////////////////////////////////////////////
// test the grains ... and then turn them off
// ... they're all on the left channel ... good!
//Synth("someGrains");
////////////////////////////////////////////////
​
// make 2 groups, the 1st for sources and the 2nd for effects
(
~source = Group.head(s);
~effects = Group.tail(s);
~allpass = Group.tail(s)
)
​
// place grains into the delay ... source is on the left and delayed source is on the right
(
Synth.head(~source, "someGrains");
Synth.head(~effects, "aDelay");
Synth.head(~allpass, "allpass");
)


~source.set(\delay, 0.25);

~source.reboot;

~source.resume;


x.free;
~source.free;

s.quit()