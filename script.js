window.onload = function () {

function makeChart(id, label) {
  return new Chart(document.getElementById(id), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: label,
        data: [],
        borderWidth: 2,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false
    }
  });
}

const hrChart = makeChart("hrChart", "Heart Rate");
const spo2Chart = makeChart("spo2Chart", "SpO₂");
const tempChart = makeChart("tempChart", "Temperature");
const stepsChart = makeChart("stepsChart", "Steps");

function update(chart, value) {
  chart.data.labels.push("");
  chart.data.datasets[0].data.push(value);

  if (chart.data.labels.length > 10) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }

  chart.update();
}

let alertActive = false;

setInterval(() => {

  let hr = Math.floor(Math.random()*40)+60;
  let spo2 = Math.floor(Math.random()*5)+95;
  let temp = (Math.random()*2+36).toFixed(1);
  let steps = Math.floor(Math.random()*200);

  if (Math.random() > 0.7) {
    hr = 140;
    spo2 = 88;
    alertActive = true;
  }

  document.getElementById("hr").innerText = hr;
  document.getElementById("spo2").innerText = spo2;
  document.getElementById("temp").innerText = temp;
  document.getElementById("steps").innerText = steps;

  update(hrChart, hr);
  update(spo2Chart, spo2);
  update(tempChart, temp);
  update(stepsChart, steps);

  const bar = document.getElementById("statusBar");

  if (hr > 120 || spo2 < 92 || alertActive) {
    bar.innerText = "🚨 EMERGENCY!";
    bar.className = "status alert";
  } else {
    bar.innerText = "💚 HEALTHY";
    bar.className = "status normal";
  }

}, 2000);

window.resolveAlert = function () {
  alertActive = false;

  const bar = document.getElementById("statusBar");
  bar.innerText = "💚 HEALTHY";
  bar.className = "status normal";

  const btn = document.getElementById("alertBtn");
  btn.innerText = "✔ Done!";
  setTimeout(() => {
    btn.innerText = "Resolve Alert";
  }, 1500);
};

};