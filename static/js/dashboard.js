// Global variables
let priceChart, revenueChart;
let autoRunning = false;
let autoRunInterval;
let currentAIAction = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    initCharts();
    updateDashboard();
    setInterval(updateDashboard, 2000); // Update every 2 seconds
});

// Initialize Charts
function initCharts() {
    // Price Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Our Price',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4
            }, {
                label: 'Competitor Avg',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1' }
                }
            },
            scales: {
                y: {
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' }
                }
            }
        }
    });

    // Revenue Chart
    const revenueCtx = document.getElementById('revenueChart').getContext('2d');
    revenueChart = new Chart(revenueCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Revenue',
                data: [],
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1' }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' }
                }
            }
        }
    });
}

// Update Dashboard
async function updateDashboard() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        // Update metrics
        document.getElementById('current-price').textContent = `₹${data.current_price.toLocaleString()}`;
        document.getElementById('seats-sold').textContent = `${data.seats_sold} / ${data.total_seats}`;
        document.getElementById('load-factor').textContent = `${data.load_factor}%`;
        document.getElementById('total-revenue').textContent = `₹${data.total_revenue.toLocaleString()}`;
        document.getElementById('days-departure').textContent = data.days_to_departure;
        document.getElementById('disruption-status').textContent = data.disruption || 'None';
        
        // Update load factor progress bar
        document.getElementById('load-progress').style.width = `${data.load_factor}%`;
        
        // Update competitor prices
        updateCompetitorPrices(data.competitor_prices);
        
        // Update analytics
        updateAnalytics();
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Update Competitor Prices
function updateCompetitorPrices(prices) {
    const container = document.getElementById('competitor-prices');
    container.innerHTML = '';
    
    for (const [airline, price] of Object.entries(prices)) {
        const item = document.createElement('div');
        item.className = 'competitor-item';
        item.innerHTML = `
            <span class="competitor-name">${airline}</span>
            <span class="competitor-price">₹${Math.round(price).toLocaleString()}</span>
        `;
        container.appendChild(item);
    }
}

// Take Action
async function takeAction(action) {
    try {
        const response = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLogEntry(data.message);
            showToast(data.message, 'success');
            
            // Update charts
            await updateHistory();
            await updateDashboard();
            
            if (data.done) {
                showToast('Flight departed! Simulation complete.', 'info');
                stopAutoRun();
            }
        }
    } catch (error) {
        console.error('Error taking action:', error);
        showToast('Error taking action', 'error');
    }
}

// Get AI Recommendation
async function getAIRecommendation() {
    try {
        const response = await fetch('/api/ai_recommendation');
        const data = await response.json();
        
        currentAIAction = data.action;
        
        const actionNames = {
            'decrease_20': '📉 Decrease 20%',
            'decrease_10': '📉 Decrease 10%',
            'hold': '⏸️ Hold Price',
            'increase_10': '📈 Increase 10%',
            'increase_20': '📈 Increase 20%'
        };
        
        document.getElementById('ai-rec-content').innerHTML = `
            <div class="ai-action">${actionNames[data.action]}</div>
            <div class="ai-reason">${data.reason}</div>
            <div style="margin-top: 10px; color: #10b981;">
                Confidence: ${(data.confidence * 100).toFixed(0)}%
            </div>
        `;
        
    } catch (error) {
        console.error('Error getting AI recommendation:', error);
    }
}

// Follow AI Recommendation
async function followAI() {
    if (currentAIAction) {
        await takeAction(currentAIAction);
        await getAIRecommendation();
    } else {
        await getAIRecommendation();
        setTimeout(() => followAI(), 500);
    }
}

// Trigger Disruption
async function triggerDisruption(type) {
    try {
        const response = await fetch('/api/disruption', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: type })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLogEntry(data.message);
            showToast(data.message, 'info');
            await updateDashboard();
        }
    } catch (error) {
        console.error('Error triggering disruption:', error);
    }
}

// Auto Run
async function autoRun() {
    if (autoRunning) {
        stopAutoRun();
        return;
    }
    
    autoRunning = true;
    const btn = event.target;
    btn.textContent = '⏸️ Stop Auto Run';
    btn.style.background = '#ef4444';
    
    addLogEntry('🤖 Auto-run started - AI taking over');
    
    autoRunInterval = setInterval(async () => {
        await getAIRecommendation();
        await followAI();
    }, 3000); // Take action every 3 seconds
}

function stopAutoRun() {
    autoRunning = false;
    clearInterval(autoRunInterval);
    
    const btn = document.querySelector('.btn-secondary');
    if (btn) {
        btn.textContent = '🤖 Auto Run (AI Mode)';
        btn.style.background = '';
    }
    
    addLogEntry('⏸️ Auto-run stopped');
}

// Reset Simulation
async function resetSimulation() {
    if (!confirm('Are you sure you want to reset the simulation?')) {
        return;
    }
    
    stopAutoRun();
    
    try {
        const response = await fetch('/api/reset', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('Simulation reset successfully', 'success');
            addLogEntry('🔄 Simulation reset to initial state');
            
            // Clear charts
            priceChart.data.labels = [];
            priceChart.data.datasets[0].data = [];
            priceChart.data.datasets[1].data = [];
            priceChart.update();
            
            revenueChart.data.labels = [];
            revenueChart.data.datasets[0].data = [];
            revenueChart.update();
            
            await updateDashboard();
        }
    } catch (error) {
        console.error('Error resetting simulation:', error);
    }
}

// Update History and Charts
async function updateHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            const last20 = data.history.slice(-20);
            
            // Update price chart
            priceChart.data.labels = last20.map(h => `Day ${h.step}`);
            priceChart.data.datasets[0].data = last20.map(h => h.price);
            
            // Calculate competitor average (simplified)
            priceChart.data.datasets[1].data = last20.map(h => h.price * 0.98);
            priceChart.update();
            
            // Update revenue chart
            revenueChart.data.labels = last20.map(h => `Day ${h.step}`);
            revenueChart.data.datasets[0].data = last20.map(h => h.revenue);
            revenueChart.update();
        }
    } catch (error) {
        console.error('Error updating history:', error);
    }
}

// Update Analytics
async function updateAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        if (!data.error) {
            const panel = document.getElementById('analytics-panel');
            panel.innerHTML = `
                <div class="analytics-item">
                    <span class="analytics-label">Total Revenue</span>
                    <span class="analytics-value">₹${data.total_revenue.toLocaleString()}</span>
                </div>
                <div class="analytics-item">
                    <span class="analytics-label">Avg Price</span>
                    <span class="analytics-value">₹${data.avg_price.toLocaleString()}</span>
                </div>
                <div class="analytics-item">
                    <span class="analytics-label">Total Bookings</span>
                    <span class="analytics-value">${data.total_bookings}</span>
                </div>
                <div class="analytics-item">
                    <span class="analytics-label">Load Factor</span>
                    <span class="analytics-value">${data.load_factor}%</span>
                </div>
                <div class="analytics-item">
                    <span class="analytics-label">Revenue/Seat</span>
                    <span class="analytics-value">₹${data.revenue_per_seat.toLocaleString()}</span>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error updating analytics:', error);
    }
}

// Add Log Entry
function addLogEntry(message) {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const now = new Date();
    const time = now.toLocaleTimeString();
    
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${message}</span>
    `;
    
    log.insertBefore(entry, log.firstChild);
    
    // Keep only last 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// Show Toast Notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Initialize AI recommendation on load
setTimeout(getAIRecommendation, 1000);