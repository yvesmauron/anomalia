(() => {
    'use strict';
    const TWILIO_DOMAIN = location.host;
    const ROOM_NAME = 'tf';
    const Video = Twilio.Video;
    let videoRoom, localStream;
    const video = document.getElementById("video");

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const minConfidence = 0.2;
    const VIDEO_WIDTH = 370;
    const VIDEO_HEIGHT = 370;
    const frameRate = 20;// default 20
    

    //-- Start edited by M. Mosisch --//

    //const skaleton_color = "#33E9FF";
    var skaleton_color = "#E633FF";

    // function set_skaleton_color(){
    //   skaleton_color = document.getElementById("html5colorpicker").value;
    //   document.getElementById("bcolor").value = skaleton_color;
    // }

    // preview screen
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(vid => {
      video.srcObject = vid;
      localStream = vid;
      const intervalID = setInterval(async () => {
        try {
          estimateMultiplePoses();
        } catch (err) {
          clearInterval(intervalID)
          setErrorMessage(err.message)
        }
      }, Math.round(1000 / frameRate))
      return () => clearInterval(intervalID)
    });

    function drawPoint(y, x, r) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = skaleton_color; // "#FFFFFF";
      ctx.fill();
    }

    function drawKeypoints(keypoints) {
      for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];
        console.log(`keypoint in drawkeypoints ${keypoint}`);
        const { y, x } = keypoint.position;
        drawPoint(y, x, 3);
      }
    }

    function drawSegment(
      pair1,
      pair2,
      color,
      scale
    ) {
      ctx.beginPath();
      ctx.moveTo(pair1.x * scale, pair1.y * scale);
      ctx.lineTo(pair2.x * scale, pair2.y * scale);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    function drawSkeleton(keypoints) {
      var color = skaleton_color ; // "#FFFFFF";
      const adjacentKeyPoints = posenet.getAdjacentKeyPoints(
        keypoints,
        minConfidence
      );

      adjacentKeyPoints.forEach((keypoint) => {
        drawSegment(
          keypoint[0].position,
          keypoint[1].position,
          color,
          1,
        );
      });
    }

    //-- Tfjs Posnet - mode : single person --//
    
    /** 
    // Model MobileNetV1
    const net = await posenet.load({  
      outputStride:16,
      architecture:'MobileNetV1',
      inputResolution: { width: 370, height: 370 },
      multiplier:0.75
     });

    //Model ResNet50
    const net = await posenet.load({
      outputStride: 32,
      architecture: 'ResNet50',
      inputResolution: { width: 256, height: 200 },      
      quantBytes: 2
    });
    */

    // define MobileNetV1 in method posenet.load()
    // Models [ MobileNetV1 | ResNet50 ]
    // default posenet.load()
    // with decodingMethod [ single-person | multi-person ]
    const estimateMultiplePoses = () => {
      posenet
        .load({  
          outputStride:16,
          architecture:'MobileNetV1',
          inputResolution: { width: 370, height: 370 },
          multiplier:0.75
         })
        .then(function (net) {
          console.log("estimateMultiplePoses .... ");
          return net.estimatePoses(video, {
            decodingMethod: "single-person",
          });
        })
        .then(function (poses) {
          console.log(`got Poses ${JSON.stringify(poses)}`);
          canvas.width = VIDEO_WIDTH;
          canvas.height = VIDEO_HEIGHT;
          ctx.clearRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
          ctx.save();
          ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
          ctx.restore();
          poses.forEach(({ score, keypoints }) => {
            if (score >= minConfidence) {
              drawKeypoints(keypoints);
              drawSkeleton(keypoints);
            }
          });
        });
    };
  
    //-- End M. Mosisch --//

    //-- join Chat Room --//
    // buttons 
    const joinRoomButton = document.getElementById("button-join");
    const leaveRoomButton = document.getElementById("button-leave");
    var site = `https://${TWILIO_DOMAIN}/video-token`;
    console.log(`site ${site}`);
    joinRoomButton.onclick = () => {
      // get access token
      axios.get(`https://${TWILIO_DOMAIN}/video-token`).then(async (body) => {
        const token = body.data.token;
        console.log(token);

        Video.connect(token, { name: ROOM_NAME }).then((room) => {
          console.log(`Connected to Room ${room.name}`);
          videoRoom = room;

          room.participants.forEach(participantConnected);
          room.on("participantConnected", participantConnected);

          room.on("participantDisconnected", participantDisconnected);
          room.once("disconnected", (error) =>
            room.participants.forEach(participantDisconnected)
          );
          joinRoomButton.disabled = true;
          leaveRoomButton.disabled = false;
        });
      });
    };
    leaveRoomButton.onclick = () => {
      videoRoom.disconnect();
      console.log(`Disconnected from Room ${videoRoom.name}`);
      joinRoomButton.disabled = false;
      leaveRoomButton.disabled = true;
    };
  //-- END of CHATROOM --//
})();

const participantConnected = (participant) => {
    console.log(`Participant ${participant.identity} connected'`);

    const div = document.createElement('div');
    div.id = participant.sid;

    participant.on('trackSubscribed', track => trackSubscribed(div, track));
    participant.on('trackUnsubscribed', trackUnsubscribed);
  
    participant.tracks.forEach(publication => {
      if (publication.isSubscribed) {
        trackSubscribed(div, publication.track);
      }
    });
    document.body.appendChild(div);
}

const participantDisconnected = (participant) => {
    console.log(`Participant ${participant.identity} disconnected.`);
    document.getElementById(participant.sid).remove();
}

const trackSubscribed = (div, track) => {
    div.appendChild(track.attach());
}

const trackUnsubscribed = (track) => {
    track.detach().forEach(element => element.remove());
}
