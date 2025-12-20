// Multi-Class Dashboard JavaScript - FIXED VERSION
let econPriceChart, busPriceChart, revenueChart;
let autoRunning = false;
let autoRunInterval;
let currentAIAction = null;
let sessionStats = {
    actions: 0,
    bookings: 0,
    rewards: []
};

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Multi-class dashboard initialized');
    initCharts();
    loadAgentInfo();
    loadRoutes();  // Load available routes for dropdown
    updateDashboard();
    setInterval(updateDashboard, 2000);
    setInterval(getAIRecommendation, 5000);
});

function initCharts() {
    const chartOptions = {
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
                ticks: { 
                    color: '#cbd5e1',
                    maxRotation: 45,
                    minRotation: 0
                }, 
                grid: { color: 'rgba(203, 213, 225, 0.1)' } 
            }
        }
    };

    // Economy Price Chart
    const econCtx = document.getElementById('econPriceChart').getContext('2d');
    econPriceChart = new Chart(econCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Our Economy Price',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Business Price Chart
    const busCtx = document.getElementById('busPriceChart').getContext('2d');
    busPriceChart = new Chart(busCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Our Business Price',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // FIXED: Daily Revenue Chart (not cumulative)
    const revCtx = document.getElementById('revenueChart').getContext('2d');
    revenueChart = new Chart(revCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Economy Daily Revenue',
                data: [],
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: '#3b82f6',
                borderWidth: 2
            }, {
                label: 'Business Daily Revenue',
                data: [],
                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                borderColor: '#8b5cf6',
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                y: { 
                    stacked: true,
                    ticks: { color: '#cbd5e1' }, 
                    grid: { color: 'rgba(203, 213, 225, 0.1)' },
                    title: {
                        display: true,
                        text: 'Daily Revenue (₹)',
                        color: '#cbd5e1'
                    }
                },
                x: { 
                    stacked: true,
                    ticks: { 
                        color: '#cbd5e1',
                        maxRotation: 45,
                        minRotation: 0
                    }, 
                    grid: { color: 'rgba(203, 213, 225, 0.1)' } 
                }
            }
        }
    });
}

// NEW: Load available routes and populate dropdown
async function loadRoutes() {
    try {
        const response = await fetch('/api/routes');
        const data = await response.json();
        
        if (data.routes && data.routes.length > 0) {
            populateRouteDropdown(data.routes, data.current_route);
            console.log('🛣️ Loaded routes:', data.routes.length);
        }
    } catch (error) {
        console.error('Error loading routes:', error);
    }
}

// NEW: Populate route dropdown with scrollable options
function populateRouteDropdown(routes, currentRoute) {
    const selector = document.getElementById('route-selector');
    if (!selector) return;
    
    // Clear existing options
    selector.innerHTML = '';
    
    // Add options
    routes.forEach(route => {
        const option = document.createElement('option');
        option.value = route;
        option.textContent = route;
        if (route === currentRoute) {
            option.selected = true;
        }
        selector.appendChild(option);
    });
    
    // Add change event listener
    selector.addEventListener('change', function() {
        changeRoute(this.value);
    });
}

// NEW: Change route handler
async function changeRoute(route) {
    try {
        showToast(`🔄 Switching to route: ${route}...`, 'info');
        
        const response = await fetch('/api/change_route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ route: route })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`✅ ${data.message}`, 'success');
            
            // Reset session stats
            sessionStats = {
                actions: 0,
                bookings: 0,
                rewards: []
            };
            
            document.getElementById('actions-count').textContent = '0';
            document.getElementById('total-bookings').textContent = '0';
            document.getElementById('avg-reward').textContent = '0.00';
            
            // Clear charts
            [econPriceChart, busPriceChart, revenueChart].forEach(chart => {
                chart.data.labels = [];
                chart.data.datasets.forEach(ds => ds.data = []);
                chart.update();
            });
            
            // Update dashboard
            await updateDashboard();
            await getAIRecommendation();
            
            addLogEntry(`🛣️ Switched to route: ${route}`);
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error changing route:', error);
        showToast('❌ Error changing route', 'error');
    }
}

async function updateDashboard() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        // Update route name
        document.getElementById('route-name').textContent = data.route || 'Loading...';
        document.getElementById('days-departure').textContent = data.days_to_departure || 90;
        document.getElementById('disruption-status').textContent = data.disruption || 'None';
        
        // Economy metrics
        document.getElementById('econ-price').textContent = `₹${Math.round(data.econ_price).toLocaleString()}`;
        document.getElementById('econ-sold').textContent = `${data.econ_sold} / ${data.econ_total}`;
        document.getElementById('econ-load').textContent = `${data.econ_load_factor.toFixed(1)}%`;
        document.getElementById('econ-progress').style.width = `${Math.min(100, data.econ_load_factor)}%`;
        document.getElementById('econ-revenue').textContent = `₹${Math.round(data.econ_revenue).toLocaleString()}`;
        
        // Business metrics
        document.getElementById('bus-price').textContent = `₹${Math.round(data.bus_price).toLocaleString()}`;
        document.getElementById('bus-sold').textContent = `${data.bus_sold} / ${data.bus_total}`;
        document.getElementById('bus-load').textContent = `${data.bus_load_factor.toFixed(1)}%`;
        document.getElementById('bus-progress').style.width = `${Math.min(100, data.bus_load_factor)}%`;
        document.getElementById('bus-revenue').textContent = `₹${Math.round(data.bus_revenue).toLocaleString()}`;
        
        // Overall
        document.getElementById('total-sold').textContent = `${data.total_sold} / ${data.total_seats}`;
        document.getElementById('total-load').textContent = `${data.load_factor.toFixed(1)}%`;
        document.getElementById('total-progress').style.width = `${Math.min(100, data.load_factor)}%`;
        document.getElementById('total-revenue').textContent = `₹${Math.round(data.total_revenue).toLocaleString()}`;
        
        // Competitors
        updateCompetitorPrices('econ-competitors', data.econ_competitors);
        updateCompetitorPrices('bus-competitors', data.bus_competitors);
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

function updateCompetitorPrices(elementId, prices) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
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

async function takeAction(actionId) {
    try {
        const response = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            sessionStats.actions++;
            sessionStats.bookings += data.total_bookings;
            sessionStats.rewards.push(data.reward);
            
            document.getElementById('actions-count').textContent = sessionStats.actions;
            document.getElementById('total-bookings').textContent = sessionStats.bookings;
            
            const avgReward = sessionStats.rewards.reduce((a, b) => a + b, 0) / sessionStats.rewards.length;
            document.getElementById('avg-reward').textContent = avgReward.toFixed(2);
            
            addLogEntry(data.message);
            showToast(data.message, 'success');
            
            await updateHistory();
            await updateDashboard();
            await getAIRecommendation();
            
            if (data.done) {
                showToast('✈️ Flight departed! Simulation complete.', 'info');
                stopAutoRun();
            }
        }
    } catch (error) {
        console.error('Error taking action:', error);
        showToast('❌ Error taking action', 'error');
    }
}

async function getAIRecommendation() {
    try {
        const response = await fetch('/api/ai_recommendation');
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('ai-rec-content').innerHTML = `
                <div class="error">❌ ${data.error}</div>
            `;
            return;
        }
        
        currentAIAction = data.action;
        
        const statusBadge = data.agent_status === 'trained' 
            ? '<span class="badge-trained">✓ TRAINED</span>' 
            : '<span class="badge-untrained">⚠️ UNTRAINED</span>';
        
        document.getElementById('ai-rec-content').innerHTML = `
            ${statusBadge}
            <div class="ai-action">${data.action_name}</div>
            <div class="ai-reason">${data.reason}</div>
            <div style="margin-top: 10px; color: #10b981;">
                Confidence: ${(data.confidence * 100).toFixed(0)}%
                ${data.q_value !== undefined ? ` | Q-value: ${data.q_value.toFixed(2)}` : ''}
            </div>
            <div class="market-context">
                <small>
                    E: ₹${Math.round(data.market_context.econ_price).toLocaleString()} (${data.market_context.econ_vs_market}), ${data.market_context.econ_load.toFixed(1)}% full<br>
                    B: ₹${Math.round(data.market_context.bus_price).toLocaleString()} (${data.market_context.bus_vs_market}), ${data.market_context.bus_load.toFixed(1)}% full
                </small>
            </div>
        `;
        
    } catch (error) {
        console.error('Error getting AI recommendation:', error);
        document.getElementById('ai-rec-content').innerHTML = `
            <div class="error">❌ Error loading recommendation</div>
        `;
    }
}

async function loadAgentInfo() {
    try {
        const response = await fetch('/api/agent_info');
        const data = await response.json();
        
        console.log('🤖 RL Agent Info:', data);
        
        const statusText = data.agent_status === 'trained' 
            ? '<span style="color: #10b981;">✓ Trained</span>' 
            : '<span style="color: #f59e0b;">⚠️ Untrained</span>';
        
        document.getElementById('agent-status').innerHTML = statusText;
        document.getElementById('agent-state-size').textContent = data.state_size;
        document.getElementById('agent-action-size').textContent = data.action_size;
        document.getElementById('agent-episodes').textContent = data.episodes_trained || '0';
        
        if (!data.agent_loaded) {
            showToast('⚠️ No trained model loaded - agent using untrained policy', 'warning');
        } else {
            showToast('✓ Trained RL agent loaded successfully', 'success');
        }
    } catch (error) {
        console.error('Error loading agent info:', error);
        document.getElementById('agent-status').innerHTML = '<span style="color: #ef4444;">❌ Error</span>';
    }
}

async function followAI() {
    if (currentAIAction !== null) {
        await takeAction(currentAIAction);
    } else {
        await getAIRecommendation();
        setTimeout(() => {
            if (currentAIAction !== null) {
                takeAction(currentAIAction);
            }
        }, 500);
    }
}

async function autoRun() {
    if (autoRunning) {
        stopAutoRun();
        return;
    }
    
    autoRunning = true;
    const btn = event.target;
    btn.textContent = '⏸️ Stop Auto';
    btn.style.background = '#ef4444';
    
    addLogEntry('🤖 Auto-run started');
    showToast('🤖 Auto-run mode activated', 'info');
    
    autoRunInterval = setInterval(async () => {
        await getAIRecommendation();
        await followAI();
    }, 3000);
}

function stopAutoRun() {
    autoRunning = false;
    clearInterval(autoRunInterval);
    
    const btn = document.querySelector('.btn-auto');
    if (btn) {
        btn.textContent = '🤖 Auto Run (AI Mode)';
        btn.style.background = '';
    }
    
    addLogEntry('⏸️ Auto-run stopped');
}

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
        showToast('❌ Error triggering disruption', 'error');
    }
}

async function resetSimulation() {
    if (!confirm('Reset multi-class simulation? This will clear all progress.')) return;
    
    stopAutoRun();
    
    try {
        const response = await fetch('/api/reset', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('✅ Multi-class simulation reset', 'success');
            addLogEntry('🔄 Reset to initial state');
            
            sessionStats = {
                actions: 0,
                bookings: 0,
                rewards: []
            };
            
            document.getElementById('actions-count').textContent = '0';
            document.getElementById('total-bookings').textContent = '0';
            document.getElementById('avg-reward').textContent = '0.00';
            
            [econPriceChart, busPriceChart, revenueChart].forEach(chart => {
                chart.data.labels = [];
                chart.data.datasets.forEach(ds => ds.data = []);
                chart.update();
            });
            
            await updateDashboard();
            await getAIRecommendation();
        }
    } catch (error) {
        console.error('Error resetting:', error);
        showToast('❌ Error resetting simulation', 'error');
    }
}

// FIXED: Update history with DAILY revenue (not cumulative)
async function updateHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            const last20 = data.history.slice(-20);
            
            const labels = last20.map((h, idx) => {
                const day = h.day || (idx + 1);
                return `Day ${day}`;
            });
            
            // Economy prices
            econPriceChart.data.labels = labels;
            econPriceChart.data.datasets[0].data = last20.map(h => 
                Math.round(h.econ_price || 0)
            );
            econPriceChart.update();
            
            // Business prices
            busPriceChart.data.labels = labels;
            busPriceChart.data.datasets[0].data = last20.map(h => 
                Math.round(h.bus_price || 0)
            );
            busPriceChart.update();
            
            // FIXED: Calculate DAILY revenue from step revenue
            revenueChart.data.labels = labels;
            
            // Each history entry should have the revenue for THAT DAY
            const econDailyRev = last20.map(h => {
                // Get bookings and price for this day
                const econBookings = h.econ_bookings || 0;
                const econPrice = h.econ_price || 0;
                return Math.round(econBookings * econPrice);
            });
            
            const busDailyRev = last20.map(h => {
                const busBookings = h.bus_bookings || 0;
                const busPrice = h.bus_price || 0;
                return Math.round(busBookings * busPrice);
            });
            
            revenueChart.data.datasets[0].data = econDailyRev;
            revenueChart.data.datasets[1].data = busDailyRev;
            revenueChart.update();
        }
    } catch (error) {
        console.error('Error updating history:', error);
    }
}

function addLogEntry(message) {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const time = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${message}</span>
    `;
    
    log.insertBefore(entry, log.firstChild);
    
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}

setTimeout(getAIRecommendation, 1000);