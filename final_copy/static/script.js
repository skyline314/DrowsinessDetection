const alarmSound = document.getElementById('alarm-sound');

const THEME = {
  green: '#50e3c2',
  red: '#e24a4a',
  yellow: '#f5a623',
  gray: '#b0b0b0',
};
const startBtn = document.getElementById('startButton');
const pauseBtn = document.getElementById('pauseButton');

let statusInterval = null;

function setButtonState(isPaused) {
  startBtn.classList.toggle('disabled', !isPaused);
  pauseBtn.classList.toggle('disabled', isPaused);
}

startBtn.addEventListener('click', () => {
  if (startBtn.classList.contains('disabled')) return;

  alarmSound.play().catch((e) => console.error('Gagal memutar suara:', e));
  alarmSound.pause();

  fetch('/start_detection', { method: 'POST' });
  setButtonState(false);
  clearInterval(statusInterval);
  statusInterval = setInterval(fetchStatus, 500);
});

pauseBtn.addEventListener('click', () => {
  if (pauseBtn.classList.contains('disabled')) return;

  fetch('/pause_detection', { method: 'POST' });
  setButtonState(true);
  clearInterval(statusInterval);
  
  // Hentikan dan reset alarm saat di-pause
  alarmSound.pause();
  alarmSound.currentTime = 0;

  resetStatusUI();
});

function resetStatusUI() {
  updateStatusUI({
    left_eye: 'N/A',
    right_eye: 'N/A',
    yawn_status: 'PAUSED',
    mar: 0.0,
    pitch: 0.0,
    yaw: 0.0,
    roll: 0.0,
    alert: false,
    is_paused: true,
  });
}

function updateStatusUI(data) {
  const leftEye = document.getElementById('left_eye_status');
  const rightEye = document.getElementById('right_eye_status');
  leftEye.textContent = data.left_eye || 'N/A';
  rightEye.textContent = data.right_eye || 'N/A';
  leftEye.style.backgroundColor =
    data.left_eye === 'Terbuka'
      ? THEME.green
      : data.left_eye === 'Tertutup'
        ? THEME.red
        : THEME.gray;
  rightEye.style.backgroundColor =
    data.right_eye === 'Terbuka'
      ? THEME.green
      : data.right_eye === 'Tertutup'
        ? THEME.red
        : THEME.gray;

  const yawnStatus = document.getElementById('yawn_status');
  yawnStatus.textContent = data.yawn_status || 'PAUSED';
  yawnStatus.style.backgroundColor =
    data.yawn_status === 'MENGUAP'
      ? THEME.yellow
      : data.is_paused
        ? THEME.gray
        : THEME.green;
  document.getElementById('mar_value').textContent = (data.mar || 0).toFixed(2);

  const pitch = data.pitch || 0,
    yaw = data.yaw || 0,
    roll = data.roll || 0;
  document.getElementById('pitch_value').textContent = pitch.toFixed(1) + '°';
  document.getElementById('yaw_value').textContent = yaw.toFixed(1) + '°';
  document.getElementById('roll_value').textContent = roll.toFixed(1) + '°';
  
  const calculateBarWidth = (value) => {
    let width = (value + 90) / 1.8;
    return Math.max(0, Math.min(100, width));
  };

  document.getElementById('pitch_bar').style.width = `${calculateBarWidth(pitch)}%`;
  document.getElementById('yaw_bar').style.width = `${calculateBarWidth(yaw)}%`;
  document.getElementById('roll_bar').style.width = `${calculateBarWidth(roll)}%`;

  const alertOverlay = document.getElementById('alert-overlay');
  if (data.alert) {
    document.getElementById('alert-reason').textContent = `Penyebab: ${data.alert_reason}`;
    alertOverlay.classList.remove('hidden');

    if (alarmSound.paused) {
        alarmSound.play().catch(e => console.error("Error memutar alarm:", e));
    }
  } else {
    alertOverlay.classList.add('hidden');
    
    alarmSound.pause();
    alarmSound.currentTime = 0;
  }
}

async function fetchStatus() {
  try {
    const response = await fetch('/status');
    if (!response.ok) {
      console.error('Network response was not ok:', response.statusText);
      clearInterval(statusInterval);
      return;
    }
    const data = await response.json();
    if (data && Object.keys(data).length > 0) {
      updateStatusUI(data);
    }
  } catch (error) {
    console.error('Gagal mengambil status:', error);
    clearInterval(statusInterval);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  setButtonState(true);
  resetStatusUI();
});