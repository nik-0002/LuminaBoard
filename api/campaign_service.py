from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import uuid
import pandas as pd
from utils.data_processors import DataProcessor
from utils.metrics import MetricsCalculator

class CampaignService:
    """
    Core campaign management service
    Works with CSV data for growers, campaigns, products, and messaging
    """
    
    def __init__(self, engagement_predictor=None, trend_detector=None, llm_client=None, 
                 messaging_orchestrator=None, audio_generator=None):
        self.engagement_predictor = engagement_predictor
        self.trend_detector = trend_detector
        self.llm = llm_client
        self.messenger = messaging_orchestrator
        self.audio_gen = audio_generator
        self.campaigns = {}
        self.campaign_metrics = {}
        self.data_processor = DataProcessor()
        self.metrics_calc = MetricsCalculator()
        
        # Load datasets
        self.datasets = self.data_processor.load_all_datasets()
    
    def create_campaign(self, campaign_config: Dict) -> Dict:
        """
        Create new marketing campaign with AI optimization
        """
        campaign_id = self._generate_campaign_id()
        
        # Get target farmer segments from CSV
        target_segments = self._get_target_segments(
            campaign_config.get('region'),
            campaign_config.get('crop'),
            campaign_config.get('target_criteria', {})
        )
        
        # Generate content variants for each segment
        content_variants = self._generate_content_variants(
            crop=campaign_config.get('crop', 'wheat'),
            product=campaign_config.get('product', 'Fungicide'),
            region=campaign_config.get('region', 'Punjab'),
            segments=target_segments,
            num_variants=campaign_config.get('num_variants', 5)
        )
        
        campaign = {
            'campaign_id': campaign_id,
            'name': campaign_config.get('name', ''),
            'product': campaign_config.get('product', ''),
            'crop': campaign_config.get('crop', ''),
            'region': campaign_config.get('region', ''),
            'target_segments': len(target_segments),
            'content_variants': content_variants,
            'scheduling': campaign_config.get('scheduling', {}),
            'budget': campaign_config.get('budget', 0),
            'created_at': datetime.now().isoformat(),
            'status': 'draft',
            'performance': {
                'sent': 0,
                'delivered': 0,
                'engaged': 0,
                'converted': 0
            }
        }
        
        self.campaigns[campaign_id] = campaign
        self.campaign_metrics[campaign_id] = self._initialize_metrics()
        
        return {
            'success': True,
            'campaign_id': campaign_id,
            'campaign': campaign,
            'target_farmer_count': len(target_segments)
        }
    
    def _get_target_segments(self, region: str, crop: str, criteria: Dict = None) -> List[Dict]:
        """
        Get target farmer segments from grower CSV based on criteria
        """
        if 'growers' not in self.datasets:
            return []
        
        growers_df = self.datasets['growers'].copy()
        
        # Filter by region/state
        if region:
            growers_df = growers_df[growers_df['state'].str.contains(region, case=False, na=False)]
        
        # Filter by crop from crop calendar
        if crop:
            growers_df = growers_df[growers_df['grower_crop_calendar'].str.contains(crop, case=False, na=False)]
        
        # Apply additional criteria
        if criteria:
            if criteria.get('device_type'):
                growers_df = growers_df[growers_df['device_type'] == criteria['device_type']]
            
            if criteria.get('min_farm_size'):
                growers_df = growers_df[growers_df['grower_farm_size'] >= criteria['min_farm_size']]
            
            if criteria.get('language'):
                growers_df = growers_df[growers_df['language'] == criteria['language']]
        
        # Convert to list of dicts
        return growers_df.to_dict('records')
    
    def _generate_content_variants(self, crop: str, product: str, region: str,
                                   segments: List[Dict], num_variants: int) -> List[Dict]:
        """
        Generate multiple content variants for A/B testing
        """
        variants = []
        
        for i in range(min(num_variants, 5)):
            # Generate base message (LLM if available, else template)
            if self.llm:
                message_result = self.llm.generate_content(
                    f"Generate a marketing message variant {i+1} for {crop} farmers in {region} about {product}",
                    temperature=0.5 + (i * 0.1)
                )
                text = message_result.get('content', f"Check out our {product} for {crop}!")
            else:
                text = f"Variant {i+1}: Use {product} for better {crop} yield in {region}"
            
            variant = {
                'variant_id': i + 1,
                'type': self._determine_variant_type(i),
                'content': {
                    'text': text,
                    'has_image': i % 3 == 0,
                    'has_audio': i % 4 == 0
                },
                'predicted_engagement': self._predict_variant_engagement(i),
                'ctr_estimate': 0.05 + (i * 0.02)
            }
            variants.append(variant)
        
        return variants
    
    def _determine_variant_type(self, index: int) -> str:
        """Determine content type for variant"""
        types = ['text_only', 'text_image', 'text_audio', 'video_script', 'voice_message']
        return types[index % len(types)]
    
    def _predict_variant_engagement(self, variant_index: int) -> float:
        """Estimate engagement for each variant"""
        if self.engagement_predictor:
            return self.engagement_predictor.predict(variant_index)
        base_engagement = 0.4
        variant_boost = (variant_index % 3) * 0.1
        return min(0.95, base_engagement + variant_boost)
    
    def _initialize_metrics(self) -> Dict:
        """Initialize campaign metrics tracker"""
        return {
            'daily_metrics': {},
            'channel_performance': {},
            'segment_performance': {},
            'engagement_by_variant': {},
            'conversion_funnel': {
                'sent': 0,
                'delivered': 0,
                'opened': 0,
                'clicked': 0,
                'converted': 0
            }
        }
    
    def launch_campaign(self, campaign_id: str, farmer_list: Optional[List[Dict]] = None) -> Dict:
        """
        Launch campaign to farmer segments
        """
        if campaign_id not in self.campaigns:
            return {'success': False, 'error': 'Campaign not found'}
        
        if not farmer_list:
            return {'success': False, 'error': 'No farmers provided'}
        
        campaign = self.campaigns[campaign_id]
        campaign['status'] = 'active'
        campaign['launched_at'] = datetime.now().isoformat()
        
        # Route messages to farmers
        delivery_results = []
        for farmer in farmer_list:
            try:
                # Select best variant for farmer
                selected_variant = self._select_variant_for_farmer(campaign['content_variants'], farmer)
                
                # Generate personalized content
                content = self._personalize_content(selected_variant, farmer, campaign)
                
                # Route message if messenger available
                if self.messenger:
                    delivery = self.messenger.route_campaign_message(
                        farmer_context=farmer,
                        message_content=content,
                        campaign_id=campaign_id
                    )
                else:
                    # Simulate delivery
                    delivery = {
                        'success': True,
                        'message_id': str(uuid.uuid4()),
                        'status': 'sent',
                        'farmer_id': farmer.get('grower_id')
                    }
                
                delivery_results.append(delivery)
                campaign['performance']['sent'] += 1
            except Exception as e:
                delivery_results.append({'success': False, 'error': str(e)})
        
        return {
            'success': True,
            'campaign_id': campaign_id,
            'total_messages': len(farmer_list),
            'successful': sum(1 for d in delivery_results if d.get('success')),
            'delivery_results': delivery_results,
            'launch_time': datetime.now().isoformat()
        }
    
    def _select_variant_for_farmer(self, variants: List[Dict], farmer: Dict) -> Dict:
        """Select best variant for individual farmer"""
        variant_scores = []
        
        for variant in variants:
            score = variant['predicted_engagement']
            
            # Adjust based on farmer characteristics
            device_type = farmer.get('device_type', 'smartphone')
            if variant['type'] == 'text_audio' and device_type == 'keypad':
                score *= 1.3
            elif variant['type'] == 'video_script' and device_type == 'smartphone':
                score *= 1.2
            
            variant_scores.append(score)
        
        best_variant_idx = max(range(len(variant_scores)), key=lambda i: variant_scores[i])
        return variants[best_variant_idx]
    
    def _personalize_content(self, variant: Dict, farmer: Dict, campaign: Dict) -> Dict:
        """
        Personalize content for specific farmer
        """
        personalized = {
            'type': variant['type'],
            'text': variant['content']['text'],
            'farmer_id': farmer.get('grower_id'),
            'variant_id': variant['variant_id']
        }
        
        # Add media if needed
        if variant['content']['has_image']:
            personalized['image'] = f"/media/campaign_{campaign['campaign_id']}_image.jpg"
        
        if variant['content']['has_audio'] and self.audio_gen:
            try:
                audio_result = self.audio_gen.generate_campaign_audio(
                    personalized['text'],
                    language=farmer.get('language', 'hi')
                )
                if audio_result and audio_result.get('success'):
                    personalized['audio_url'] = audio_result.get('audio_path')
            except:
                pass
        
        return personalized
    
    def get_campaign_analytics(self, campaign_id: str, time_period: str = '7d') -> Dict:
        """
        Get comprehensive campaign analytics
        """
        if campaign_id not in self.campaigns:
            return {'success': False, 'error': 'Campaign not found'}
        
        campaign = self.campaigns[campaign_id]
        metrics = self.campaign_metrics.get(campaign_id, {})
        
        # Calculate KPIs
        delivery_rate = (campaign['performance']['delivered'] / 
                        campaign['performance']['sent']) if campaign['performance']['sent'] > 0 else 0
        engagement_rate = (campaign['performance']['engaged'] / 
                          campaign['performance']['delivered']) if campaign['performance']['delivered'] > 0 else 0
        conversion_rate = (campaign['performance']['converted'] / 
                          campaign['performance']['engaged']) if campaign['performance']['engaged'] > 0 else 0
        
        return {
            'campaign_id': campaign_id,
            'campaign_name': campaign['name'],
            'status': campaign['status'],
            'period': time_period,
            'performance_metrics': {
                'total_sent': campaign['performance']['sent'],
                'delivery_rate': round(delivery_rate * 100, 2),
                'engagement_rate': round(engagement_rate * 100, 2),
                'conversion_rate': round(conversion_rate * 100, 2),
                'conversions': campaign['performance']['converted']
            },
            'channel_breakdown': metrics.get('channel_performance', {}),
            'segment_breakdown': metrics.get('segment_performance', {}),
            'top_variants': self._get_top_variants(campaign_id),
            'daily_trend': metrics.get('daily_metrics', {})
        }
    
    def _get_top_variants(self, campaign_id: str, top_n: int = 3) -> List[Dict]:
        """Get top performing variants"""
        if campaign_id not in self.campaigns:
            return []
        
        campaign = self.campaigns[campaign_id]
        variants_with_perf = []
        
        for variant in campaign['content_variants']:
            perf = self.campaign_metrics[campaign_id].get('engagement_by_variant', {}).get(
                variant['variant_id'], 0
            )
            variants_with_perf.append({
                'variant_id': variant['variant_id'],
                'type': variant['type'],
                'engagement_score': perf
            })
        
        return sorted(variants_with_perf, key=lambda x: x['engagement_score'], reverse=True)[:top_n]
    
    def get_dataset_insights(self) -> Dict:
        """
        Get insights from loaded CSV datasets
        """
        insights = {
            'total_growers': 0,
            'total_campaigns': 0,
            'total_retailers': 0,
            'states': [],
            'crops': [],
            'products': []
        }
        
        if 'growers' in self.datasets:
            growers_df = self.datasets['growers']
            insights['total_growers'] = len(growers_df)
            insights['states'] = growers_df['state'].unique().tolist()
        
        if 'campaigns' in self.datasets:
            campaigns_df = self.datasets['campaigns']
            insights['total_campaigns'] = len(campaigns_df)
            insights['crops'] = campaigns_df['campaign_crop'].unique().tolist()
            insights['products'] = campaigns_df['campaign_product'].unique().tolist()
        
        if 'retailers' in self.datasets:
            insights['total_retailers'] = len(self.datasets['retailers'])
        
        return insights
    
    def _generate_campaign_id(self) -> str:
        """Generate unique campaign ID"""
        return f"camp_{uuid.uuid4().hex[:8]}"
