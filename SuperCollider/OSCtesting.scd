n = NetAddr.new("127.0.0.1", 57120);    // create the NetAddr

s.boot;
// create the OSCFunc
//o = OSCFunc({ arg msg, time, addr, recvPort; [msg, time, addr, recvPort].postln; }, '/rand', n);
//o;


o = (
OSCFunc.new(
	\getMovenetData,
	{
		"recieved".postln;
	},
	'/randomNum',

);

)
o.free;

(
OSCFunc({
    "received".postln;
}, '/randomNum').fix;
)


(
OSCFunc({
    |msg|  // msg contains the entire OSC message data
    var number = msg[1];  // Access the first argument of the message, which is the random number
    "Received number: ".post;  // Print a label
    number.postln;  // Print the number itself
}, '/randomNum').fix;
)

n = NetAddr("127.0.0.1", 57120);
n.sendMsg('/randomNum',466);


//o.free;    // remove the OSCFunc when you are done.


s.quit;