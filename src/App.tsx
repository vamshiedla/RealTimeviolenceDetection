import React, { useState, useRef, useEffect } from 'react';
import { Camera, AlertTriangle, Play, Square, Volume2 } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import emailjs from '@emailjs/browser';

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [violencePercentage, setViolencePercentage] = useState(0);
  const [isAlertActive, setIsAlertActive] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const alertTimeoutRef = useRef<NodeJS.Timeout>();
  const alertAudioRef = useRef<HTMLAudioElement>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const addLog = (message: string) => {
    setLogs(prev => [`${new Date().toLocaleTimeString()} - ${message}`, ...prev]);
  };

  const loadModel = async () => {
    try {
      const model = await tf.loadLayersModel('https://teachablemachine.withgoogle.com/models/muzbGbrap/model.json');
      modelRef.current = model;
      addLog('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      addLog('Error loading violence detection model');
    }
  };

  useEffect(() => {
    loadModel();
    // Get user's location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error) => {
          console.error('Error getting location:', error);
        }
      );
    }
  }, []);

  const captureSnapshot = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        return canvasRef.current.toDataURL('image/jpeg');
      }
    }
    return null;
  };

  const sendEmailAlert = async (snapshot: string | null) => {
    try {
      const templateParams = {
        to_email: 'security@example.com',
        location: location ? `${location.latitude}, ${location.longitude}` : 'Location not available',
        timestamp: new Date().toLocaleString(),
        snapshot: snapshot || 'No snapshot available',
        violence_level: Math.round(violencePercentage)
      };

      const response = await emailjs.send(
        'YOUR_SERVICE_ID',
        'YOUR_TEMPLATE_ID',
        templateParams,
        'YOUR_PUBLIC_KEY'
      );

      addLog('Email alert sent successfully');
    } catch (error) {
      console.error('Error sending email:', error);
      addLog('Failed to send email alert');
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsRunning(true);
      addLog('Surveillance started');
    } catch (error) {
      console.error('Error accessing camera:', error);
      addLog('Error accessing camera');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setViolencePercentage(0);
    addLog('Surveillance stopped');
  };

  const triggerAlert = async () => {
    setIsAlertActive(true);
    if (alertAudioRef.current) {
      alertAudioRef.current.play();
    }
    
    // Capture snapshot and send notifications
    const snapshot = captureSnapshot();
    await sendEmailAlert(snapshot);
    
    addLog('⚠️ ALERT: High violence detected!');
    setTimeout(() => setIsAlertActive(false), 5000);
  };

  const predictViolence = async () => {
    if (!videoRef.current || !modelRef.current) return;

    // Convert video frame to tensor
    const videoTensor = tf.browser.fromPixels(videoRef.current)
      .resizeNearestNeighbor([224, 224]) // Resize to model's expected size
      .expandDims()
      .toFloat()
      .div(255.0);

    // Get prediction
    const prediction = await modelRef.current.predict(videoTensor) as tf.Tensor;
    const probabilityArray = await prediction.data();
    
    // Cleanup tensors
    videoTensor.dispose();
    prediction.dispose();

    // Violence probability is the second class (index 1)
    const violenceProbability = probabilityArray[1] * 100;
    setViolencePercentage(violenceProbability);

    if (violenceProbability > 95) {
      if (!alertTimeoutRef.current) {
        alertTimeoutRef.current = setTimeout(() => {
          triggerAlert();
          alertTimeoutRef.current = undefined;
        }, 1000); // Reduced to 1 second for high violence
      }
    } else {
      if (alertTimeoutRef.current) {
        clearTimeout(alertTimeoutRef.current);
        alertTimeoutRef.current = undefined;
      }
    }
  };

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      predictViolence();
    }, 1000);

    return () => {
      clearInterval(interval);
      if (alertTimeoutRef.current) {
        clearTimeout(alertTimeoutRef.current);
      }
    };
  }, [isRunning]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Panel - Project Information */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h1 className="text-2xl font-bold mb-4">Real-Time Violence Detection in Live CCTV</h1>
              
              <h2 className="text-xl font-semibold mb-2">Project Abstract</h2>
              <p className="text-gray-300 mb-4">
                This system utilizes machine learning to detect violence in real-time through the laptop's webcam.
                Using a trained model from Teachable Machine, it continuously monitors the video feed and sends
                immediate alerts if violent behavior is detected for a sustained period.
              </p>

              <h2 className="text-xl font-semibold mb-2">Dataset/Model Information</h2>
              <ul className="list-disc list-inside text-gray-300 mb-4">
                <li>Model: Teachable Machine (Image Classification)</li>
                <li>Classes: Violent, Non-violent</li>
                <li>
                  <a 
                    href="https://teachablemachine.withgoogle.com/models/muzbGbrap/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300"
                  >
                    View Model
                  </a>
                </li>
              </ul>

              <h2 className="text-xl font-semibold mb-2">Technologies Used</h2>
              <ul className="list-disc list-inside text-gray-300 mb-4">
                <li>Language: Python + JavaScript</li>
                <li>Frontend: React + Tailwind CSS</li>
                <li>Model Integration: Teachable Machine Web API</li>
                <li>Camera Access: HTML5 getUserMedia()</li>
                <li>Alert System: Audio + Visual + Email Notifications</li>
                <li>Location Tracking: Browser Geolocation API</li>
              </ul>

              <h2 className="text-xl font-semibold mb-2">Team Members</h2>
              <table className="w-full text-gray-300">
                <thead>
                  <tr>
                  <th className="text-left">Name</th>
                    <th className="text-left">Roll Number</th>
                    
                  </tr>
                </thead>
                <tbody>
                  <tr>
                  <td>Vamshi Edla (TL)</td>
                    <td>23915A7206</td>
                    
                  </tr>
                  <tr>
                    <td>B. Harinath</td>
                    <td>22911A7208</td>
                    
                  </tr>
                  <tr>
                    
                    <td>P. Keerthana</td>
                    <td>22911A7246</td>
                  </tr>
                  <tr>
                    
                    <td>T. Sharon Peter</td>
                    <td>22911A7246</td>
                  </tr>
                </tbody>
              </table>

              <h2 className="text-xl font-semibold mt-4 mb-2">Guide</h2>
              <p className="text-gray-300">Guide Name: Divya Sarabudla</p>
              {/* <p className="text-gray-300">Guide Signature: ___________</p> */}
            </div>
          </div>

          {/* Right Panel - Violence Detection Dashboard */}
          <div className="lg:col-span-8 space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center">
                  <Camera className="mr-2" /> Live Surveillance Feed - Violence Detection
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={isRunning ? stopCamera : startCamera}
                    className={`px-4 py-2 rounded-lg flex items-center ${
                      isRunning
                        ? 'bg-red-600 hover:bg-red-700'
                        : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    {isRunning ? (
                      <>
                        <Square className="mr-2" size={16} />
                        Stop Surveillance
                      </>
                    ) : (
                      <>
                        <Play className="mr-2" size={16} />
                        Start Surveillance
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Video Feed */}
              <div className="relative aspect-video mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full bg-black rounded-lg"
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
                {isAlertActive && (
                  <div className="absolute top-0 left-0 right-0 bg-red-600 text-white p-2 rounded-t-lg flex items-center justify-center">
                    <AlertTriangle className="mr-2" />
                    ⚠️ Violence Detected! Alert Sent to Security
                  </div>
                )}
              </div>

              {/* Violence Detection Meter */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Violence Detection Level</span>
                  <span className="text-sm font-medium">{Math.round(violencePercentage)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div
                    className="bg-red-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${violencePercentage}%` }}
                  />
                </div>
              </div>

              {/* Location Information */}
              {location && (
                <div className="mb-4 text-sm text-gray-300">
                  <p>📍 Location: {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}</p>
                </div>
              )}

              {/* Logs Panel */}
              <div>
                <h3 className="text-lg font-semibold mb-2 flex items-center">
                  <Volume2 className="mr-2" /> Activity Log
                </h3>
                <div className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto">
                  {logs.map((log, index) => (
                    <div key={index} className="text-sm text-gray-300 mb-1">
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alert Sound */}
      <audio ref={alertAudioRef}>
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg" />
      </audio>
    </div>
  );
}

export default App;