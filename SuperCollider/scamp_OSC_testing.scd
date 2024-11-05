s.boot;

//Server.local.boot;
//Server.local.quit;
NetAddr.langPort;

OSCFunc.trace(hideStatusMSG: True)




SynthDef("simpleSine",{
	arg freq =440, amp =1;
	var sig;
	sig = SinOsc.ar(freq)*amp;
}).play

SynthDef("simpleSine",{
	arg freq =440, amp =1;
	var sig;
	sig = SinOsc.ar(freq)*amp;
	Out.ar(0, sig),Out.ar(1, sig);
}).play


ScampUtils.instrumentFromSynthDef(
	SynthDef("simpleSine",{
		arg freq =440, volume =0.7, gate=0;
		var sig = SinOsc.ar(freq)*volume;
		var env = EnvGate(gate:gate, fadeTime:0.1);
		Out.ar(0, sig*env);
	})
)



z.set(\freq, 220);
z.set(\amp, 0.5);
z.free




s.quit;