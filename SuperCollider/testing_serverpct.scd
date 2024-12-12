
s.boot()


(
z = {
	arg noiseHZ=8;
	var freq, amp, sig;
	freq = LFNoise0.kr(noiseHZ).exprange(200,1000);
	amp = LFNoise1.kr(12).exprange(0.02,1);
	sig = SinOsc.ar(freq) * amp;
}.play;
)

z.free;

s.quit()