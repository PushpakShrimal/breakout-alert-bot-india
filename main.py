# COMPREHENSIVE BREAKOUT ALERT BOT - INDIA
# Live monitoring for NSE/BSE stocks, options, futures

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class BreakoutAlertBot:
    def __init__(self):
        self.telegram_chat_id = "1113702328"
        
        # ALL MAJOR STOCKS (161+ stocks)
        self.all_stocks = [
            # Nifty 50
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'ASIANPAINT', 'LT',
            'AXISBANK', 'MARUTI', 'NESTLEIND', 'HCLTECH', 'BAJFINANCE',
            'WIPRO', 'ULTRACEMCO', 'TATAMOTORS', 'ONGC', 'NTPC', 'POWERGRID',
            'TECHM', 'TITAN', 'SUNPHARMA', 'COALINDIA', 'INDUSINDBK',
            'ADANIENT', 'JSWSTEEL', 'TATASTEEL', 'GRASIM', 'HINDALCO',
            'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'BRITANNIA', 'DIVISLAB',
            'DRREDDY', 'EICHERMOT', 'HEROMOTOCO', 'CIPLA', 'APOLLOHOSP',
            'BAJAJ-AUTO', 'TATACONSUM', 'BPCL', 'LTIM', 'ADANIPORTS',
            
            # Next 50 + Midcap
            'ADANIGREEN', 'ADANIPOWER', 'AMBUJACEM', 'BANDHANBNK', 'BERGEPAINT',
            'BIOCON', 'BOSCHLTD', 'CANBK', 'CHOLAFIN', 'COLPAL', 'CONCOR',
            'COFORGE', 'DABUR', 'DALBHARAT', 'DEEPAKNTR', 'DIVI', 'DLF',
            'GAIL', 'GODREJCP', 'HAVELLS', 'HDFCAMC', 'ICICIPRULI', 'IDEA',
            'INDIGO', 'IOC', 'IRCTC', 'JINDALSTEL', 'JUBLFOOD', 'LICHSGFIN',
            'LUPIN', 'MARICO', 'MCDOWELL-N', 'MFSL', 'MOTHERSON', 'MPHASIS',
            'MRF', 'NAUKRI', 'NMDC', 'OFSS', 'PAGEIND', 'PEL', 'PERSISTENT',
            'PETRONET', 'PIDILITIND', 'PIIND', 'PNB', 'POLYCAB', 'RAMCOCEM',
            'SAIL', 'SHREECEM', 'SIEMENS', 'TORNTPHARM', 'VOLTAS', 'ZEEL',
            
            # Additional high-volume stocks
            'ABCAPITAL', 'ACC', 'ALKEM', 'ASHOKLEY', 'ASTRAL', 'AUROPHARMA',
            'BALKRISIND', 'BATAINDIA', 'BEL', 'BHARATFORG', 'BHEL', 'CADILAHC',
            'CANFINHOME', 'CHAMBLFERT', 'CROMPTON', 'CUB', 'CUMMINSIND',
            'DELTACORP', 'DIXON', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK',
            'FORTIS', 'GLENMARK', 'GMRINFRA', 'GNFC', 'GODREJPROP',
            'GRANULES', 'GUJGASLTD', 'HINDPETRO', 'HONAUT', 'IDFCFIRSTB',
            'IEX', 'IGL', 'INDHOTEL', 'INDUSTOWER', 'IOB', 'IRFC',
            'JKCEMENT', 'JSWENERGY', 'JUBILANT', 'KAJARIACER', 'KEI',
            'LALPATHLAB', 'LAURUSLABS', 'MANAPPURAM', 'MAZDOCK', 'METROPOLIS'
        ]
        
        # Convert to NSE format
        self.nse_symbols = [f"NSE_{stock}" for stock in self.all_stocks]
        self.indices = ['NSE_NIFTY', 'NSE_BANKNIFTY', 'NSE_FINNIFTY']
        self.watchlist = self.nse_symbols + self.indices
        
        # Data storage
        self.price_history = {}
        self.last_alerts = {}
        
        print(f"🚀 Breakout Alert Bot Initialized")
        print(f"📊 Monitoring {len(self.watchlist)} instruments")
        print(f"📱 Telegram Chat: {self.telegram_chat_id}")
    
    def calculate_indicators(self, prices, volumes):
        """Calculate comprehensive technical indicators"""
        if len(prices) < 30:
            return None
        
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['price'].ewm(span=12).mean()
        exp2 = df['price'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        # Volume analysis
        vol_sma_20 = df['volume'].rolling(20).mean()
        vol_sma_10 = df['volume'].rolling(10).mean()
        
        # Support/Resistance
        resistance_20 = df['price'].rolling(20).max()
        support_20 = df['price'].rolling(20).min()
        resistance_50 = df['price'].rolling(50).max() if len(prices) >= 50 else resistance_20
        support_50 = df['price'].rolling(50).min() if len(prices) >= 50 else support_20
        
        # Momentum
        momentum_5 = ((df['price'].iloc[-1] / df['price'].iloc[-6]) - 1) * 100 if len(prices) >= 6 else 0
        momentum_10 = ((df['price'].iloc[-1] / df['price'].iloc[-11]) - 1) * 100 if len(prices) >= 11 else 0
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'macd_signal': signal.iloc[-1] if not signal.empty else 0,
            'macd_crossover': self.detect_macd_crossover(macd, signal),
            'volume_ratio_20': df['volume'].iloc[-1] / vol_sma_20.iloc[-1] if vol_sma_20.iloc[-1] > 0 else 1,
            'volume_ratio_10': df['volume'].iloc[-1] / vol_sma_10.iloc[-1] if vol_sma_10.iloc[-1] > 0 else 1,
            'resistance_20': resistance_20.iloc[-1] if not resistance_20.empty else prices[-1],
            'support_20': support_20.iloc[-1] if not support_20.empty else prices[-1],
            'resistance_50': resistance_50.iloc[-1] if not resistance_50.empty else prices[-1],
            'support_50': support_50.iloc[-1] if not support_50.empty else prices[-1],
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'current_price': prices[-1]
        }
    
    def detect_macd_crossover(self, macd, signal):
        """Detect MACD crossover"""
        if len(macd) < 2:
            return 'NONE'
        
        if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            return 'BULLISH'
        elif macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            return 'BEARISH'
        else:
            return 'NONE'
    
    def calculate_momentum_score(self, indicators):
        """Calculate momentum score (0-100)"""
        score = 0
        
        # RSI momentum (25 points)
        rsi = indicators['rsi']
        if rsi > 70:
            score += 25
        elif rsi > 60:
            score += 20
        elif rsi > 50:
            score += 10
        elif rsi < 30:
            score += 20  # Oversold bounce potential
        elif rsi < 40:
            score += 15
        
        # MACD momentum (25 points)
        if indicators['macd_crossover'] == 'BULLISH':
            score += 20
        if indicators['macd'] > indicators['macd_signal']:
            score += 5
        
        # Volume surge (25 points)
        vol_ratio = max(indicators['volume_ratio_10'], indicators['volume_ratio_20'])
        if vol_ratio > 3:
            score += 25
        elif vol_ratio > 2:
            score += 20
        elif vol_ratio > 1.5:
            score += 15
        elif vol_ratio > 1.2:
            score += 10
        
        # Price momentum (25 points)
        momentum = max(abs(indicators['momentum_5']), abs(indicators['momentum_10']))
        if momentum > 5:
            score += 25
        elif momentum > 3:
            score += 20
        elif momentum > 2:
            score += 15
        elif momentum > 1:
            score += 10
        
        return min(score, 100)
    
    def generate_signals(self, symbol, indicators, momentum_score):
        """Generate trading signals"""
        signals = []
        current_price = indicators['current_price']
        
        # STRONG BREAKOUT
        if (momentum_score > 85 and
            indicators['volume_ratio_20'] > 2.5 and
            current_price > indicators['resistance_20'] * 0.998 and
            indicators['rsi'] > 65):
            
            signals.append({
                'type': 'STRONG_BREAKOUT',
                'symbol': symbol.replace('NSE_', ''),
                'price': current_price,
                'confidence': 95,
                'target_1': current_price * 1.025,
                'target_2': current_price * 1.04,
                'stop_loss': current_price * 0.985,
                'momentum_score': momentum_score,
                'reason': f"Exceptional momentum ({momentum_score}), massive volume ({indicators['volume_ratio_20']:.1f}x), resistance break"
            })
        
        # MOMENTUM BREAKOUT
        elif (momentum_score > 70 and
              indicators['rsi'] > 60 and
              indicators['macd_crossover'] == 'BULLISH' and
              indicators['volume_ratio_20'] > 1.8):
            
            signals.append({
                'type': 'MOMENTUM_BREAKOUT',
                'symbol': symbol.replace('NSE_', ''),
                'price': current_price,
                'confidence': 85,
                'target_1': current_price * 1.02,
                'target_2': current_price * 1.035,
                'stop_loss': current_price * 0.99,
                'momentum_score': momentum_score,
                'reason': f"Strong momentum ({momentum_score}), bullish MACD crossover, volume surge"
            })
        
        # OVERSOLD BOUNCE
        elif (indicators['rsi'] < 30 and
              current_price < indicators['support_20'] * 1.01 and
              indicators['volume_ratio_20'] > 1.5 and
              momentum_score > 40):
            
            signals.append({
                'type': 'OVERSOLD_BOUNCE',
                'symbol': symbol.replace('NSE_', ''),
                'price': current_price,
                'confidence': 80,
                'target_1': (indicators['support_20'] + indicators['resistance_20']) / 2,
                'target_2': indicators['resistance_20'],
                'stop_loss': current_price * 0.975,
                'momentum_score': momentum_score,
                'reason': f"Oversold RSI ({indicators['rsi']:.1f}), support level, volume support"
            })
        
        # BREAKDOWN
        elif (momentum_score < 25 and
              current_price < indicators['support_20'] * 1.002 and
              indicators['volume_ratio_20'] > 2 and
              indicators['rsi'] < 45):
            
            signals.append({
                'type': 'BREAKDOWN',
                'symbol': symbol.replace('NSE_', ''),
                'price': current_price,
                'confidence': 90,
                'target_1': current_price * 0.98,
                'target_2': current_price * 0.965,
                'stop_loss': current_price * 1.015,
                'momentum_score': momentum_score,
                'reason': f"Weak momentum ({momentum_score}), support break, volume confirmation"
            })
        
        return signals
    
    def generate_option_strategy(self, signal):
        """Generate option trading strategy"""
        symbol = signal['symbol']
        spot_price = signal['price']
        
        # Calculate strikes
        if spot_price > 2000:
            interval = 100
        elif spot_price > 1000:
            interval = 50
        else:
            interval = 25
        
        atm = round(spot_price / interval) * interval
        
        strategies = []
        
        if signal['type'] in ['STRONG_BREAKOUT', 'MOMENTUM_BREAKOUT']:
            otm_strike = atm + interval
            strategies.append({
                'strategy': 'CALL_BUYING',
                'symbol': f"{symbol}{otm_strike}CE",
                'strike': otm_strike,
                'premium_estimate': spot_price * 0.015,
                'target': spot_price * 0.03,
                'stop_loss': spot_price * 0.008,
                'max_profit_potential': '150-200%',
                'risk_reward': '1:2.5'
            })
            
        elif signal['type'] == 'OVERSOLD_BOUNCE':
            strategies.append({
                'strategy': 'ATM_CALL_BUYING',
                'symbol': f"{symbol}{atm}CE",
                'strike': atm,
                'premium_estimate': spot_price * 0.02,
                'target': spot_price * 0.025,
                'stop_loss': spot_price * 0.01,
                'max_profit_potential': '100-150%',
                'risk_reward': '1:1.5'
            })
            
        elif signal['type'] == 'BREAKDOWN':
            otm_put = atm - interval
            strategies.append({
                'strategy': 'PUT_BUYING',
                'symbol': f"{symbol}{otm_put}PE",
                'strike': otm_put,
                'premium_estimate': spot_price * 0.015,
                'target': spot_price * 0.03,
                'stop_loss': spot_price * 0.008,
                'max_profit_potential': '150-200%',
                'risk_reward': '1:2.5'
            })
        
        return strategies
    
    def generate_futures_analysis(self, signal):
        """Generate futures analysis"""
        spot_price = signal['price']
        
        # Estimate futures price
        futures_price = spot_price * 1.002  # Simplified
        
        analysis = {
            'contract': f"{signal['symbol']}_FUT",
            'spot_price': spot_price,
            'futures_price': futures_price,
            'margin_required': spot_price * 0.12,  # ~12% margin
            'lot_size': 1000 if signal['symbol'] in ['NIFTY', 'BANKNIFTY'] else 500
        }
        
        if signal['type'] in ['STRONG_BREAKOUT', 'MOMENTUM_BREAKOUT']:
            analysis['recommendation'] = {
                'action': 'BUY',
                'entry': futures_price,
                'target_1': futures_price * 1.025,
                'target_2': futures_price * 1.04,
                'stop_loss': futures_price * 0.985,
                'leverage': '8-10x'
            }
        elif signal['type'] == 'BREAKDOWN':
            analysis['recommendation'] = {
                'action': 'SELL',
                'entry': futures_price,
                'target_1': futures_price * 0.975,
                'target_2': futures_price * 0.96,
                'stop_loss': futures_price * 1.015,
                'leverage': '8-10x'
            }
        
        return analysis
    
    def format_comprehensive_alert(self, signal, option_strategies, futures_analysis):
        """Format comprehensive alert message"""
        message = f"""🚨 {signal['type']} - {signal['symbol']}

📊 STOCK ANALYSIS:
Price: ₹{signal['price']:.2f}
Momentum Score: {signal['momentum_score']}/100 🔥
Confidence: {signal['confidence']}%
Target 1: ₹{signal['target_1']:.2f}
Target 2: ₹{signal['target_2']:.2f}
Stop Loss: ₹{signal['stop_loss']:.2f}

💡 Reason: {signal['reason']}

📈 OPTION STRATEGY:"""
        
        if option_strategies:
            opt = option_strategies[0]
            message += f"""
Strategy: {opt['strategy']}
Symbol: {opt['symbol']}
Premium: ₹{opt['premium_estimate']:.2f}
Target: ₹{opt['target']:.2f}
Stop Loss: ₹{opt['stop_loss']:.2f}
Profit Potential: {opt['max_profit_potential']}
Risk-Reward: {opt['risk_reward']}"""
        
        if futures_analysis and 'recommendation' in futures_analysis:
            fut = futures_analysis['recommendation']
            message += f"""

🔮 FUTURES TRADE:
Contract: {futures_analysis['contract']}
Action: {fut['action']}
Entry: ₹{fut['entry']:.2f}
Target 1: ₹{fut['target_1']:.2f}
Target 2: ₹{fut['target_2']:.2f}
Margin: ₹{futures_analysis['margin_required']:.0f}
Leverage: {fut['leverage']}"""
        
        message += f"""

🕒 Time: {datetime.now().strftime('%H:%M:%S')}
📊 NSE | ⚡ Live Analysis | 🎯 High Probability Setup"""
        
        return message
    
    def send_telegram_alert(self, message):
        """Send alert via Telegram (placeholder - integrate with actual API)"""
        try:
            # This would be replaced with actual Telegram API call
            print(f"📱 TELEGRAM ALERT SENT:")
            print(message)
            print("-" * 50)
            return True
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")
            return False
    
    def should_send_alert(self, symbol):
        """Check if alert should be sent (5-minute cooldown)"""
        if symbol in self.last_alerts:
            time_diff = datetime.now() - self.last_alerts[symbol]
            return time_diff.total_seconds() > 300
        return True
    
    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def simulate_market_data(self):
        """Simulate market data for testing (replace with actual Groww API)"""
        # This is for testing - replace with actual Groww API calls
        simulated_data = {}
        
        for symbol in self.watchlist[:5]:  # Test with first 5 symbols
            base_price = 1000 + hash(symbol) % 2000
            price_change = np.random.normal(0, 20)
            volume_multiplier = np.random.uniform(0.8, 3.0)
            
            simulated_data[symbol] = {
                'ltp': base_price + price_change,
                'volume': int(100000 * volume_multiplier),
                'change': price_change,
                'change_percent': (price_change / base_price) * 100
            }
        
        return simulated_data
    
    def analyze_symbol(self, symbol, current_data):
        """Analyze individual symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        # Add current data
        self.price_history[symbol].append({
            'price': current_data['ltp'],
            'volume': current_data['volume'],
            'timestamp': datetime.now()
        })
        
        # Keep last 100 data points
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Need minimum 30 data points
        if len(self.price_history[symbol]) < 30:
            return None
        
        # Extract data
        prices = [item['price'] for item in self.price_history[symbol]]
        volumes = [item['volume'] for item in self.price_history[symbol]]
        
        # Calculate indicators
        indicators = self.calculate_indicators(prices, volumes)
        
        if not indicators:
            return None
        
        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(indicators)
        
        # Generate signals
        signals = self.generate_signals(symbol, indicators, momentum_score)
        
        return signals
    
    def run_monitoring_cycle(self):
        """Single monitoring cycle"""
        try:
            print(f"🔍 Scanning at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get market data (replace with actual Groww API)
            market_data = self.simulate_market_data()
            
            if not market_data:
                print("❌ No market data received")
                return
            
            alerts_sent = 0
            
            # Analyze each symbol
            for symbol, data in market_data.items():
                signals = self.analyze_symbol(symbol, data)
                
                if signals and self.should_send_alert(symbol):
                    for signal in signals:
                        # Generate option strategies
                        option_strategies = self.generate_option_strategy(signal)
                        
                        # Generate futures analysis
                        futures_analysis = self.generate_futures_analysis(signal)
                        
                        # Format comprehensive alert
                        alert_message = self.format_comprehensive_alert(
                            signal, option_strategies, futures_analysis
                        )
                        
                        # Send alert
                        if self.send_telegram_alert(alert_message):
                            self.last_alerts[symbol] = datetime.now()
                            alerts_sent += 1
            
            print(f"✅ Cycle complete - {alerts_sent} alerts sent")
            
        except Exception as e:
            print(f"❌ Error in monitoring cycle: {e}")
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Comprehensive Breakout Alert Bot - India")
        print(f"📊 Monitoring {len(self.watchlist)} instruments")
        print(f"📱 Telegram Chat: {self.telegram_chat_id}")
        print(f"⏰ Market Hours: 9:15 AM - 3:30 PM IST")
        print("=" * 60)
        
        # Add some initial data for testing
        for symbol in self.watchlist[:5]:
            base_price = 1000 + hash(symbol) % 2000
            for i in range(35):
                price = base_price + np.random.normal(0, 10) + i * 0.5
                volume = 100000 + np.random.normal(0, 20000)
                
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                self.price_history[symbol].append({
                    'price': price,
                    'volume': max(volume, 50000),
                    'timestamp': datetime.now() - timedelta(minutes=35-i)
                })
        
        while True:
            try:
                if self.is_market_open():
                    self.run_monitoring_cycle()
                    time.sleep(30)  # Check every 30 seconds
                else:
                    print("📴 Market closed - waiting...")
                    time.sleep(300)  # Wait 5 minutes when closed
                    
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    bot = BreakoutAlertBot()
    bot.run()