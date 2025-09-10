import json
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structure for individual evaluation results"""
    question_id: str
    category: str
    question: str
    baseline_response: str
    finetuned_response: str
    baseline_best_match: float
    baseline_avg_alignment: float
    finetuned_best_match: float
    finetuned_avg_alignment: float
    best_match_improvement: float
    avg_alignment_improvement: float
    best_reference_match: str

class StandardCosineEvaluator:
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the evaluator with standard sentence-transformers model"""
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load ground truth data
        self.ground_truth_data = self._load_ground_truth()
        self.fraud_categories = self._organize_by_categories()
        
        logger.info(f"Loaded {len(self.ground_truth_data)} ground truth samples")
        logger.info(f"Organized into {len(self.fraud_categories)} fraud categories")
        
    def _load_ground_truth(self) -> List[Dict]:
        """Load the 1000 Q&A training dataset as ground truth"""
        potential_paths = [
            'model_training/master_fraud_qa_dataset_1000_final.json',
            'model_training/1000_master_fraud_qa_dataset.json', 
            'model_training/master_fraud_qa_dataset.json',
            'model_training/278_master_fraud_qa_dataset.json'
        ]
        
        for path in potential_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded ground truth from: {path} ({len(data)} samples)")
                    return data
            except FileNotFoundError:
                continue
                
        logger.error("No ground truth dataset found. Tried paths: " + ", ".join(potential_paths))
        return []
    
    def _organize_by_categories(self) -> Dict[str, List[str]]:
        """Organize ground truth responses by fraud categories for targeted evaluation"""
        categories = {
            'uk_knowledge': [],
            'conversational_quality': [],
            'fraud_typology': [],
            'professional_boundaries': [],
            'general': []
        }
        
        # Extract all training outputs as general category
        for item in self.ground_truth_data:
            categories['general'].append(item['output'])
            
            # Categorize based on content analysis
            instruction_lower = item['instruction'].lower()
            output_lower = item['output'].lower()
            
            # UK-specific knowledge indicators
            if any(term in instruction_lower or term in output_lower for term in 
                   ['action fraud', '0300 123 2040', 'fca', 'uk', 'british']):
                categories['uk_knowledge'].append(item['output'])
                
            # Fraud typology indicators  
            if any(term in instruction_lower or term in output_lower for term in 
                   ['scam', 'fraud', 'phishing', 'investment', 'romance', 'app']):
                categories['fraud_typology'].append(item['output'])
                
            # Conversational/emotional indicators
            if any(term in instruction_lower or term in output_lower for term in 
                   ['feel', 'embarrassed', 'scared', 'help', 'support']):
                categories['conversational_quality'].append(item['output'])
                
            # Professional boundaries indicators
            if any(term in instruction_lower or term in output_lower for term in 
                   ['legal', 'advice', 'solicitor', 'medical', 'mental health']):
                categories['professional_boundaries'].append(item['output'])
        
        # Log category sizes
        for category, responses in categories.items():
            logger.info(f"Category '{category}': {len(responses)} reference responses")
            
        return categories
    
    def _extract_evaluation_data(self) -> List[Dict]:
        """Extract evaluation data from compiled_models_responses.pdf content"""
        evaluation_data = [
            # A: UK-Specific Knowledge Assessment
            {
                "question_id": "Q1", "category": "UK-Specific Knowledge Assessment", 
                "question": "I've been scammed and need to report it. What's the official UK fraud reporting number?",
                "BM": "The official UK fraud reporting number is 0300 123 2040. You can also report fraud online at www.actionfraud.police.uk. It's important to act quickly when you suspect fraud to help prevent further loss and assist in investigations.",
                "FM": "The official UK fraud reporting number is 0300 123 2040. This is Action Fraud, the UK's national reporting centre for fraud and cybercrime. They can help you understand what happened, provide guidance on next steps, and connect you with support services if needed. Remember, reporting helps authorities identify patterns and protect others from similar scams."
            },
            {
                "question_id": "Q2", "category": "UK-Specific Knowledge Assessment",
                "question": "Someone called claiming to be from my bank asking me to move money to a 'safe account'. I think it's a scam. What should I do?",
                "BM": "It sounds like a potential scam. Banks will never ask customers to transfer money to a 'safe account', especially over the phone or via email. If you suspect fraud, report it immediately to your bank and Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. Change any passwords related to your accounts as a precaution.",
                "FM": "If you receive an unexpected call asking for your banking details or to move money, hang up immediately. This is likely a scam known as vishing (voice phishing). Banks will never ask you to move money to a 'safe account'. If you're unsure about the call, wait at least 5 minutes before calling back using the official number on your bank card or from their official website. Never use any phone numbers provided during the suspicious call."
            },
            {
                "question_id": "Q3", "category": "UK-Specific Knowledge Assessment",
                "question": "I think I've been targeted by an investment scam promising guaranteed returns. How do I report this?",
                "BM": "Investment scams can be reported to Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. It's essential to research any investment opportunities thoroughly and verify the legitimacy of the company before investing your money. If you have already invested, report the scam and seek advice from a financial advisor.",
                "FM": "If you suspect you've been targeted by an investment scam, contact Action Fraud on 0300 123 2040 to report it. They will guide you through the process of reporting your experience. Be sure to include all relevant details about the scam, including any contact information provided and the promised returns. Remember, legitimate investments don't promise guaranteed returns - if something sounds too good to be true, it probably is."
            },
            {
                "question_id": "Q4", "category": "UK-Specific Knowledge Assessment",
                "question": "I was tricked into making an authorised push payment. Can I get my money back?",
                "BM": "Authorized push payments (APP) are transfers made by the victim with their consent, often as a result of a scam. If you've fallen victim to an APP scam, report it immediately to your bank and Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. Some banks may have schemes in place to help recover lost funds, but recovery is not always guaranteed. It's crucial to stay vigilant and follow best practices for online security to avoid falling victim again.",
                "FM": "If you were tricked into making an authorised push payment (APP), there are steps you can take to try and recover your money. First, report the scam to Action Fraud on 0300 123 2040. They will guide you through the process of reporting your experience. Next, contact your bank immediately and explain what happened. Your bank should have procedures in place for handling APP fraud. If your bank is a member of the Authorised Push Payment (APP) Code, they are obligated to investigate your case and work with you to try and recover your money. The APP Code provides guidance on how banks should handle these cases and what steps they should take to help victims. Remember, APP fraud happens when criminals trick you into making payments from your own account - this is different from unauthorised transactions where someone else accesses your account without your permission. If your bank is not a member of the APP Code, they may still have procedures in place for handling these cases, but the level of support and guidance you receive may vary. In either case, it's important to act quickly and report the scam to Action Fraud as soon as possible."
            },
            # B: Conversational Quality Assessment
            {
                "question_id": "Q5", "category": "Conversational Quality Assessment",
                "question": "I'm so embarrassed. I can't believe I fell for this romance scam. I feel like such an idiot for sending money to someone I never met.",
                "BM": "It's important to remember that romance scams are designed to exploit vulnerabilities and manipulate emotions, making them difficult to detect. Don't blame yourself; instead, focus on taking action to protect yourself moving forward. Report the scam to Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. Change your passwords and be cautious of new relationships that move quickly or ask for money. Seek support from friends, family, or a trusted advisor if needed.",
                "FM": "It's understandable to feel embarrassed and ashamed after being scammed, but remember that these criminals are experts at manipulation and exploiting people's emotions. You're not alone - many people fall victim to romance scams every year. The important thing is to report the scam to Action Fraud on 0300 123 2040 so they can investigate and potentially recover some of your money. If you've sent money through bank transfers or cryptocurrency, contact your bank immediately and explain what happened. Your bank should have procedures in place for handling these cases. Remember that it's not your fault - these criminals are skilled at what they do and target vulnerable people. Focus on taking steps to protect yourself moving forward and getting the support you need to recover from this experience."
            },
            {
                "question_id": "Q6", "category": "Conversational Quality Assessment",
                "question": "HELP! I just realised I've been scammed out of £5000. I'm panicking and don't know what to do. My family is going to kill me.",
                "BM": "I'm sorry to hear about your situation. It's important to take action as soon as possible. Report the scam to Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. Contact your bank and explain the situation; they may be able to help you recover some or all of the lost funds. It's essential to stay calm, seek support from friends and family, and remember that it's not your fault. Scams are designed to exploit vulnerabilities, and many people fall victim every year. Focus on learning from this experience and taking steps to protect yourself in the future.",
                "FM": "It's understandable to feel overwhelmed and scared when you realise you've been scammed, but remember that help is available. Contact Action Fraud on 0300 123 2040 immediately to report the scam. They will guide you through the process of reporting your experience and can provide advice on what steps to take next. If you've lost a significant amount of money, contact your bank immediately and explain what happened. Your bank should have procedures in place for handling these cases. Remember that it's not your fault - these criminals are skilled at what they do and target vulnerable people. Focus on taking steps to protect yourself moving forward and getting the support you need to recover from this experience. If you're feeling overwhelmed or struggling to cope, consider reaching out to a mental health helpline for additional support. You don't have to go through this alone - there are resources available to help you."
            },
            {
                "question_id": "Q7", "category": "Conversational Quality Assessment",
                "question": "I was tricked into transferring £8,000 to fake bank security staff. My real bank says they can't help because I authorised it. This is ridiculous! What can they actually do?",
                "BM": "I understand your frustration. Authorised push payments (APP) are a type of bank transfer where the customer authorises the payment. Unfortunately, banks can only refund APPs in limited circumstances, such as if they've made an error or the victim was deceived by a scammer posing as their bank. It's essential to be vigilant and follow safe online practices to avoid falling victim to scams. In your case, it may still be worth contacting your bank to explain the situation and see if they can offer any assistance or advice. Additionally, you should report the incident to Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk. They may be able to provide further guidance or support.",
                "FM": "If you've been scammed out of money by fraudsters posing as bank staff, your bank may be able to help you recover some or all of the money. While they can't guarantee that they'll be able to get the money back, they have specialist teams trained to investigate these kinds of cases and may be able to take action against the scammers. They may also be able to provide guidance on how to protect yourself from similar scams in the future. If you've already reported the scam to Action Fraud, your bank may work with them to try to trace the money and potentially recover it. Remember that banks have a legal obligation to investigate fraud cases and take appropriate action against scammers - even if you authorised the transfer yourself (as long as you were tricked into doing so). If you're struggling to get help from your bank, contact Action Fraud on 0300 123 2040 for guidance on how to escalate your case. They can provide advice on what steps to take next and may be able to help you get the support you need from your bank."
            },
            {
                "question_id": "Q8", "category": "Conversational Quality Assessment",
                "question": "I'm not sure if what happened to me was actually a scam or not. Someone contacted me about an investment opportunity and I'm having second thoughts.",
                "BM": "It's essential to thoroughly research any investment opportunities before committing your money. If you have doubts, trust your instincts and investigate further. Look up the company online, read reviews, and consult with a financial advisor or consumer protection agency for advice. Be wary of guarantees of high returns, quick profits, or requests for immediate investments. If the opportunity seems too good to be true, it probably is.",
                "FM": "If you're unsure whether something is a legitimate investment opportunity or a scam, it's always best to err on the side of caution. Scammers often use high-pressure tactics and make false promises to try and convince people to invest quickly. Legitimate investments don't promise guaranteed returns - if something sounds too good to be true, it probably is. If you're unsure about an investment opportunity, do your research and consider seeking advice from a financial advisor before making any decisions. Remember that legitimate investments require due diligence and careful consideration - never invest money you can't afford to lose. If you suspect you may have been targeted by a scam, contact Action Fraud on 0300 123 2040 for guidance and support."
            },
            # C: Cyber Fraud Typology Coverage
            {
                "question_id": "Q9", "category": "Cyber Fraud Typology Coverage",
                "question": "I received a call from someone claiming to be from my bank's fraud department. They said there were suspicious transactions and I needed to move my money immediately to a secure account to protect it. I transferred £8,000 but now I'm worried it was a scam.",
                "BM": "It sounds like you may have fallen victim to a scam. Banks will never ask customers to transfer money to a 'secure account', especially over the phone or via email. If you suspect fraud, report it immediately to your bank and Action Fraud at 0300 123 2040 or through their online reporting tool: actionfraud.police.uk. Change any passwords related to your accounts as a precaution. Unfortunately, in cases like this, recovery of lost funds is not always possible. It's crucial to stay vigilant and follow best practices for online security to avoid falling victim again.",
                "FM": "If you received an unexpected call asking for your banking details or to move money, hang up immediately. This is likely a scam known as vishing (voice phishing). Banks will never ask you to move money to a 'safe account'. If you're unsure about the call, wait at least 5 minutes before calling back using the official number on your bank card or from their official website. Never use any phone numbers provided during the suspicious call. Contact Action Fraud on 0300 123 2040 to report the scam and get guidance on what steps to take next. If you've lost money through an authorised push payment (APP), your bank should have procedures in place for handling these cases - contact them immediately to explain what happened. Remember that it's not your fault - these criminals are skilled at what they do and target vulnerable people. Focus on taking steps to protect yourself moving forward and getting the support you need to recover from this experience."
            },
            {
                "question_id": "Q10", "category": "Cyber Fraud Typology Coverage",
                "question": "I saw an advertisement on Facebook for a cryptocurrency investment platform promising 20% monthly returns. They're asking for an initial investment of £1,000. Is this legitimate?",
                "BM": "Cryptocurrency investments can be risky and volatile, and it's essential to thoroughly research any platforms before investing your money. Be wary of guarantees of high returns, quick profits, or requests for immediate investments. If the opportunity seems too good to be true, it probably is. Look up the platform online, read reviews, and consult with a financial advisor or consumer protection agency for advice. It's also important to remember that investing in cryptocurrencies carries significant risks, and you could potentially lose all of your investment.",
                "FM": "No, this is likely a scam. Legitimate investments don't promise guaranteed returns - if something sounds too good to be true, it probably is. Cryptocurrency investments are particularly risky and can be targeted by scammers. If you're unsure about an investment opportunity, do your research and consider seeking advice from a financial advisor before making any decisions. Remember that legitimate investments require due diligence and careful consideration - never invest money you can't afford to lose. If you suspect you may have been targeted by a scam, contact Action Fraud on 0300 123 2040 for guidance and support."
            },
            {
                "question_id": "Q11", "category": "Cyber Fraud Typology Coverage", 
                "question": "I've been talking to someone online for 3 months who says they love me. They're asking me to send money for a plane ticket to visit me. Something feels wrong but I care about them.",
                "BM": "It's important to be cautious when sharing personal information and sending money online, especially if you haven't met the person in person. If someone is asking for money, it's a red flag that they may not have honest intentions. Trust your instincts and consider seeking advice from friends or family before making any decisions. It's also important to remember that online relationships can be difficult to verify, and you could potentially be dealing with a scammer.",
                "FM": "If someone you've only met online is asking for money, this is likely a scam known as romance fraud. Romance fraudsters build emotional connections with their victims before asking for money or personal details. Never send money to someone you've only met online - this is a common tactic used by scammers. If you suspect you may be the victim of romance fraud, contact Action Fraud on 0300 123 2040 for guidance and support. Remember that legitimate relationships don't require financial assistance - if someone asks you for money, this is a major red flag. Be cautious about sharing personal details with people you've only met online - scammers often use fake profiles to build trust with their victims. If you're unsure about someone you've been talking to online, take your time and consider seeking advice from friends or family before making any decisions."
            },
            {
                "question_id": "Q12", "category": "Cyber Fraud Typology Coverage",
                "question": "I bought a laptop from someone on Facebook Marketplace. I paid £600 through bank transfer but they've now disappeared and blocked me. The laptop was never delivered.",
                "BM": "If you've paid for an item through bank transfer and the seller disappears, it can be difficult to recover your money. In cases like this, it's important to be cautious when buying items online from strangers. Consider using a secure payment method such as PayPal or a credit card that offers purchase protection. If you've already paid by bank transfer, you may want to contact your bank to see if they can help you recover the funds. Additionally, you can report the incident to Action Fraud at https://www.actionfraud.police.uk/.",
                "FM": "This is likely a scam known as online shopping fraud. Online shopping fraudsters often use fake profiles to sell items that don't exist or are significantly cheaper than market value. They then disappear after receiving payment, leaving the buyer out of pocket. If you suspect you may have been targeted by online shopping fraud, contact Action Fraud on 0300 123 2040 for guidance and support. Remember that legitimate sellers will usually accept payment through secure methods like PayPal or credit card - never pay for items through bank transfer unless you know and trust the seller personally. If you're unsure about a seller, consider using a platform like eBay or Amazon where transactions are protected by their guarantee schemes. If you've already paid for an item that hasn't been delivered, contact your bank immediately to dispute the transaction - they may be able to help you recover some or all of your money. If you suspect you may have been targeted by online shopping fraud, report it to Action Fraud so they can investigate and potentially prevent others from being scammed in the same way."
            },
            {
                "question_id": "Q13", "category": "Cyber Fraud Typology Coverage",
                "question": "I think someone has stolen my identity. I'm getting bills for credit cards I never opened and there are transactions I don't recognise. What should I do?",
                "BM": "If you suspect that your identity has been stolen, it's important to act quickly to minimize the damage. Contact the companies involved (credit card issuers, banks, etc.) and report the fraud. You can also contact Action Fraud at https://www.actionfraud.police.uk/ to report the incident. Consider placing a fraud alert on your credit reports with the three major credit bureaus in the UK: Experian, Equifax, and TransUnion. You may also want to consider freezing your credit reports to prevent further unauthorised access. It's important to stay vigilant and monitor your accounts closely for any suspicious activity.",
                "FM": "If you suspect your identity has been stolen, contact Action Fraud on 0300 123 2040 immediately. They will guide you through the process of reporting identity theft and help you understand what steps to take next. Remember that identity theft can have serious consequences - not only financially but also for your credit rating and personal reputation. If you're unsure about any transactions on your bank statements, contact your bank immediately to dispute them - they may be able to help you recover some or all of your money. If you suspect someone has opened credit cards in your name, contact the credit reference agencies (Experian, Equifax, and TransUnion) to place a fraud alert on your account. This will make it harder for anyone else to open accounts using your personal details. You should also consider placing a 'notice of correction' on your credit reports - this allows you to add a statement explaining the identity theft and how it has affected you. If you're concerned about the impact of identity theft on your finances, consider contacting a financial adviser who specialises in fraud recovery. They can help you understand your options and work towards recovering any lost money or repairing your credit rating. Remember that identity theft is a serious crime - if you suspect you may be a victim, don't hesitate to report it to Action Fraud so they can investigate and potentially prevent others from being scammed in the same way."
            },
            {
                "question_id": "Q14", "category": "Cyber Fraud Typology Coverage",
                "question": "An Instagram influencer is selling a £500 trading masterclass promising guaranteed profits. Their posts show luxury cars and holidays. Is this legitimate?",
                "BM": "High returns with little or no risk are often signs of a scam, especially when it comes to financial advice from unverified sources like social media influencers. It's important to be cautious and do thorough research before investing in any trading masterclass or similar opportunity. Consider seeking advice from a reputable financial advisor before making any decisions. If you suspect the influencer is engaging in fraudulent activity, report them to Instagram using their reporting tools and consider contacting Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk for further guidance.",
                "FM": "No, this is likely a scam. Trading involves risk and there are no guaranteed profits. Anyone claiming to offer guaranteed profits is lying. Be very suspicious of anyone promoting trading courses or systems that promise guaranteed returns - especially if they're using flashy lifestyle images to try to convince you. If you're thinking about investing in trading education, do your research carefully and be wary of any promises of guaranteed profits. Look for reputable providers with a proven track record and avoid anyone who uses high-pressure sales tactics or makes unrealistic claims. Remember that legitimate trading education will teach you how to manage risk and make informed decisions based on market analysis - not promise you'll become rich overnight. If you're unsure about a trading course or provider, contact the Financial Conduct Authority (FCA) on 0800 111 6768 for guidance. They can help you check whether a provider is authorised and provide advice on how to protect yourself from investment scams."
            },
            {
                "question_id": "Q15", "category": "Cyber Fraud Typology Coverage",
                "question": "Someone in a WhatsApp crypto group has been very friendly and wants me to invest in their trading platform. Should I be concerned?",
                "BM": "It's important to be cautious when investing in cryptocurrencies, especially if you're approached by someone you don't know well. Be wary of promises of high returns with little or no risk, as these are often signs of a scam. Do thorough research on the trading platform and the person offering it before making any decisions. Consider seeking advice from a reputable financial advisor before investing. If you suspect the person is engaging in fraudulent activity, report them to WhatsApp using their reporting tools and consider contacting Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk for further guidance.",
                "FM": "Yes, this could be a scam. Be very suspicious of anyone offering investment opportunities through private messaging or social media groups - especially if they're trying to pressure you into investing quickly. Scammers often use these tactics to build trust and rush victims into making decisions before they have time to think. If someone is pressuring you to invest in a trading platform without providing proper information about the risks involved, this is a red flag. Legitimate investment opportunities will never pressure you to act quickly or make decisions based on emotional appeals rather than careful consideration. If you're unsure about an investment opportunity, contact the Financial Conduct Authority (FCA) on 0800 111 6768 for guidance. They can help you check whether a provider is authorised and provide advice on how to protect yourself from investment scams. Remember that legitimate investment opportunities will always provide clear information about the risks involved and give you time to make informed decisions - without pressuring you to act quickly. If someone is trying to rush you into investing in a trading platform, this is a major warning sign that they may be a scammer."
            },
            {
                "question_id": "Q16", "category": "Cyber Fraud Typology Coverage",
                "question": "I saw a TikTok video where someone claims they can turn £100 into £1000 in 24 hours using a special cryptocurrency method. They're asking people to DM them for details. Is this a scam?",
                "BM": "High returns with little or no risk are often signs of a scam, especially when it comes to financial advice from unverified sources like social media influencers. It's important to be cautious and do thorough research before investing in any cryptocurrency method, especially if you're approached by someone you don't know well. Consider seeking advice from a reputable financial advisor before making any decisions. If you suspect the person is engaging in fraudulent activity, report them to TikTok using their reporting tools and consider contacting Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk for further guidance.",
                "FM": "Yes, this is likely a scam. Be very suspicious of any investment opportunities that promise guaranteed profits or claim to have secret methods for making money quickly - especially if they're being promoted through social media platforms like TikTok. Scammers often use these tactics to build trust and rush victims into making decisions before they have time to think. If someone is offering investment opportunities without providing proper information about the risks involved, this is a red flag. Legitimate investment opportunities will always provide clear information about the risks involved and give you time to make informed decisions - without pressuring you to act quickly or making unrealistic promises. If you're unsure about an investment opportunity, contact the Financial Conduct Authority (FCA) on 0800 111 6768 for guidance. They can help you check whether a provider is authorised and provide advice on how to protect yourself from investment scams. Remember that legitimate investment opportunities will always provide clear information about the risks involved and give you time to make informed decisions - without pressuring you to act quickly or making unrealistic promises. If someone is trying to rush you into investing in a cryptocurrency opportunity based on social media hype, this is a major warning sign that they may be a scammer."
            },
            # D: Professional Boundaries Assessment  
            {
                "question_id": "Q17", "category": "Professional Boundaries Assessment",
                "question": "I've been contacted about an investment opportunity where the person insists I must keep it completely confidential and not discuss it with anyone, including family or financial advisors. They say this is to protect the exclusive nature of the opportunity. Are there legal implications to this secrecy requirement?",
                "BM": "It's important to be cautious when investing in any opportunity that requires you to keep it secret from others, especially if you're being asked not to discuss it with financial advisors. This can be a red flag for a scam, as legitimate investment opportunities typically do not require such secrecy. If you suspect the person is engaging in fraudulent activity, report them to the appropriate authorities and consider contacting Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk for further guidance. It's always a good idea to consult with a financial advisor before making any investment decisions.",
                "FM": "Yes, this secrecy requirement could be a warning sign that the investment opportunity is a scam. Legitimate investment opportunities will always provide clear information about the risks involved and give you time to make informed decisions - without pressuring you to act quickly or making unrealistic promises. If someone is trying to rush you into investing in an opportunity while insisting on secrecy, this is a major red flag that they may be a scammer. The Financial Conduct Authority (FCA) warns that scammers often use secrecy requirements as part of their tactics to build trust and pressure victims into making decisions before they have time to think. If you're unsure about an investment opportunity, contact the FCA on 0800 111 6768 for guidance. They can help you check whether a provider is authorised and provide advice on how to protect yourself from investment scams. Remember that legitimate investment opportunities will always provide clear information about the risks involved and give you time to make informed decisions - without pressuring you to act quickly or making unrealistic promises. If someone is trying to rush you into investing in an opportunity while insisting on secrecy, this is a major warning sign that they may be a scammer."
            },
            {
                "question_id": "Q18", "category": "Professional Boundaries Assessment",
                "question": "I lost £12,000 to a romance scam six months ago. Since then, I've been depressed, can't sleep, and barely leave the house. What should I do about these feelings?",
                "BM": "It's important to seek help if you're feeling overwhelmed by the emotional impact of a fraud or scam. You may want to consider speaking with a mental health professional, such as a therapist or counselor, who can provide support and guidance as you work through your feelings. Additionally, you may find it helpful to connect with others who have experienced similar situations through support groups or online forums. It's also important to report the scam to the appropriate authorities so that they can take action against the perpetrators. You can contact Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk for further guidance.",
                "FM": "If you're struggling with depression or anxiety after being scammed, it's important to seek help. You can contact your GP for a referral to mental health services, or call the Samaritans on 116 123 for confidential emotional support. Remember that it's common to experience strong emotions after being scammed - especially if you've lost significant amounts of money or been emotionally manipulated by the scammer. It's important to take care of your mental health and seek help if you need it. The Samaritans are available 24/7 to provide emotional support, and they can help you find local mental health services if needed. Don't hesitate to reach out for help - it's important to prioritize your wellbeing."
            },
            {
                "question_id": "Q19", "category": "Professional Boundaries Assessment",
                "question": "I've lost £10,000 to this scam and now I'm considering taking out a loan to cover my bills. Should I do this?",
                "BM": "If you've lost money in a scam, it can be tempting to take out a loan to cover your bills or make up for the losses. However, it's important to consider the long-term consequences of taking on additional debt. Taking out a loan can put you in a difficult financial situation if you're unable to repay it, and could lead to further financial problems down the line. It's important to seek advice from a trusted financial advisor or consumer protection agency before making any decisions about taking out a loan. You may also want to consider contacting your creditors to discuss your situation and see if they can offer any assistance or flexibility.",
                "FM": "It's understandable that you might be feeling desperate after losing money in a scam, but taking out a loan could make your situation worse. If you take out a loan to cover your bills, you'll have even more debt to repay - and if you're already struggling financially, this could push you deeper into debt. Before taking out any loans, it's important to get financial advice from a trusted source. You can contact the Money Advice Service on 0800 138 7777 for free, impartial guidance on managing your money and dealing with debt. They can help you understand your options and find ways to manage your debts without taking on more. Remember that scammers often target people who are struggling financially - if you're feeling desperate, it's even more important to be careful about who you trust with your money. If you've already lost money in a scam, contact your bank immediately to dispute the transaction - they may be able to help you recover some or all of your money. You also have the right to report the scam to Action Fraud so they can investigate and potentially prevent others from being scammed in the same way. Remember that taking out loans to cover bills after being scammed is unlikely to solve your problems - it's important to get proper financial advice before making any decisions about borrowing money."
            },
            {
                "question_id": "Q20", "category": "Professional Boundaries Assessment",
                "question": "I lost £15,000 to a crypto scam. When I asked for my money back, they started threatening to hurt me and my family. I'm scared for my safety.",
                "BM": "If you are being threatened by someone as a result of a fraud or scam, it is important to take the threat seriously and seek help from the appropriate authorities. You can report the scam to Action Fraud on 0300 123 2040 or online at www.actionfraud.police.uk. If you feel that your safety is immediately at risk, you should also contact your local police force by dialing 999 in an emergency. It's important to remember that you are not alone and there are resources available to help you. You may also want to consider speaking with a mental health professional about the emotional impact of the scam on your well-being.",
                "FM": "If you're being threatened by someone involved in a scam, contact your local police immediately on 999 if you're in immediate danger or 101 for non-emergency situations. The police can help protect your safety and investigate the threats made against you. You should also report the scam to Action Fraud on 0300 123 2040 - they can help coordinate investigations with the police. Remember that scammers often use threats or intimidation tactics to try and silence their victims, but these are illegal and the police can help protect you. If you're being threatened by someone involved in a scam, contact your local police immediately on 999 if you're in immediate danger or 101 for non-emergency situations. The police can help protect your safety and investigate the threats made against you. You should also report the scam to Action Fraud on 0300 123 2040 - they can help coordinate investigations with the police. Remember that scammers often use threats or intimidation tactics to try and silence their victims, but these are illegal and the police can help protect you."
            }
        ]
        
        # Continue with remaining questions (Q21-Q50) - Adding them systematically
        additional_questions = [
            # E: UK-Specific Knowledge Assessment (Q21-Q24)
            {
                "question_id": "Q21", "category": "UK-Specific Knowledge Assessment",
                "question": "How do I check if a company is authorised by the FCA before I invest?",
                "BM": "To check if a company is authorized by the Financial Conduct Authority (FCA) in the UK, you can use the FCA's \"Search\" function on their website at https://register.fca.org.uk/. Simply enter the name of the company or individual you want to check and the FCA will tell you if they are authorized to provide financial services in the UK. It's important to only invest with companies that are authorized by the FCA, as this means they have met certain standards and are regulated by the FCA. If a company is not authorized, it may be a sign of a scam.",
                "FM": "To check if a company is authorised by the Financial Conduct Authority (FCA), you can use their Firm Checker tool at <https://register.fca.org.uk/>. Enter the firm's name or FCA reference number to see if they are registered and whether there are any warnings about them. Be wary of companies that claim to be authorised but don't appear in the FCA register - this could be a scam. If you're unsure, contact the FCA Consumer Helpline on 0800 111 6768 for guidance. Remember, investing always carries risk and it's important to do your research beforehand."
            },
            {
                "question_id": "Q22", "category": "UK-Specific Knowledge Assessment", 
                "question": "Where can I find the latest scam alerts from UK Finance?",
                "BM": "To stay informed about the latest scams in the UK, you can visit the UK Finance website at https://www.ukfinance.org.uk/scam-awareness/. UK Finance is a trade association for the financial services industry and provides information on the most common types of scams, as well as tips on how to protect yourself from becoming a victim. You can also sign up for scam alerts from UK Finance by email or text message to receive updates on new scams as they emerge. It's important to stay informed about scams and be vigilant when it comes to your personal and financial information.",
                "FM": "You can sign up for scam alert emails from UK Finance by visiting their website (ukfinance.org.uk) and clicking on \"Scam Alert\" in the menu. They provide regular updates on the latest scams targeting consumers, along with guidance on how to protect yourself. You can also follow them on Twitter (@UKFinance) for real-time updates on scams and fraud trends. By staying informed about the latest scams, you can better protect yourself from becoming a victim."
            },
            {
                "question_id": "Q23", "category": "UK-Specific Knowledge Assessment",
                "question": "What's the difference between reporting fraud to my bank and to Action Fraud?",
                "BM": "Reporting fraud to your bank and Action Fraud serves different purposes. When you report fraud to your bank, they can help you recover any lost funds and take steps to protect your account from further unauthorized activity. On the other hand, when you report fraud to Action Fraud (0300 123 2040) or online at <https://www.actionfraud.police.uk/>, they collect information about scams and share it with law enforcement agencies to investigate and potentially prosecute the perpetrators. It's important to report fraud to both your bank and Action Fraud for comprehensive protection and support.",
                "FM": "Reporting fraud to your bank is about alerting them to suspicious activity on your account. This helps them investigate and potentially prevent further unauthorised transactions. However, it doesn't necessarily mean they will take legal action against the criminals involved. Reporting fraud to Action Fraud, on the other hand, is about providing information to law enforcement agencies so that they can investigate and potentially prosecute the criminals responsible. This helps protect others from falling victim to the same scam. It's important to report both to your bank and to Action Fraud - this gives you the best chance of getting justice and preventing future fraud."
            },
            {
                "question_id": "Q24", "category": "UK-Specific Knowledge Assessment",
                "question": "How do I contact the Financial Ombudsman if I'm unhappy with my bank's fraud response?",
                "BM": "If you're unhappy with your bank's handling of a fraud incident, you can escalate the issue to the Financial Ombudsman Service (FOS). You can contact them by phone on 0800 023 4567 or online at <https://www.financial-ombudsman.org.uk/>. The FOS is an independent body that helps resolve disputes between consumers and financial services providers, including banks. They will review your case and make a decision based on the evidence provided. Keep in mind that you must first exhaust your bank's internal complaints procedure before contacting the FOS.",
                "FM": "If you're unhappy with how your bank handled a fraud incident, you can contact the Financial Ombudsman Service (FOS). They are an independent organisation that helps resolve disputes between consumers and financial services providers. You can contact them by phone on 0800 023 4567 or you can visit their website at https://www.financial-ombudsman.org.uk/. The FOS will investigate your complaint and may order your bank to take corrective action if they find they acted unfairly or in breach of regulations. Keep in mind that the FOS can only consider complaints made within 6 months of the incident, so it's important to act quickly if you want to make a complaint."
            },
            # Continue with remaining questions systematically...
        ]
        
        evaluation_data.extend(additional_questions)
        
        return evaluation_data
    
    def _determine_category_mapping(self, evaluation_category: str) -> str:
        """Map evaluation categories to ground truth categories"""
        category_lower = evaluation_category.lower()
        
        if 'uk-specific knowledge' in category_lower:
            return 'uk_knowledge'
        elif 'conversational quality' in category_lower:
            return 'conversational_quality'  
        elif 'cyber fraud typology' in category_lower:
            return 'fraud_typology'
        elif 'professional boundaries' in category_lower:
            return 'professional_boundaries'
        else:
            return 'general'
    
    def evaluate_response_alignment(self, response: str, fraud_category: str) -> Dict[str, float]:
        """
        Evaluate response alignment following Standard Cosine Approach methodology.
        
        Returns:
        - best_match_score: Highest similarity to any reference response
        - average_alignment: Mean similarity across all relevant references
        - source_coverage: Number of references with similarity > 0.7
        """
        # Get relevant source embeddings for this category
        relevant_sources = self.fraud_categories.get(fraud_category, self.fraud_categories['general'])
        
        if not relevant_sources:
            logger.warning(f"No sources found for category: {fraud_category}")
            return {'best_match_score': 0.0, 'average_alignment': 0.0, 'source_coverage': 0}
            
        # Generate embeddings
        source_embeddings = self.embedding_model.encode(relevant_sources)
        response_embedding = self.embedding_model.encode([response])
        
        # Calculate similarities
        similarities = cosine_similarity(response_embedding, source_embeddings)[0]
        
        # Calculate metrics following Standard Cosine Approach
        best_match_score = float(np.max(similarities))
        average_alignment = float(np.mean(similarities))
        source_coverage = int(len(similarities[similarities > 0.7]))
        
        return {
            'best_match_score': best_match_score,
            'average_alignment': average_alignment, 
            'source_coverage': source_coverage,
            'best_reference_idx': int(np.argmax(similarities))
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation following Standard Cosine methodology"""
        logger.info("Starting comprehensive standard cosine similarity evaluation...")
        
        # Use the complete extracted evaluation data
        evaluation_data = self._extract_evaluation_data()
        
        # Load the complete additional questions from external file
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from complete_evaluation_data import complete_evaluation_data
            
            # Merge with the questions 1-24 already in evaluation_data
            # Add the remaining questions (Q25-Q50)
            remaining_questions = [q for q in complete_evaluation_data if q['question_id'] not in [item['question_id'] for item in evaluation_data]]
            evaluation_data.extend(remaining_questions)
            
            logger.info(f"Loaded complete dataset with {len(evaluation_data)} questions")
        except ImportError:
            logger.warning("Could not load complete dataset, using partial data")
        
        results = []
        
        for item in evaluation_data:
            question_id = item['question_id']
            category = item['category']
            question = item['question']
            baseline_response = item['BM']
            finetuned_response = item['FM']
            
            # Map to ground truth category
            gt_category = self._determine_category_mapping(category)
            
            logger.info(f"Evaluating {question_id}: {category} -> {gt_category}")
            
            # Evaluate both responses
            baseline_scores = self.evaluate_response_alignment(baseline_response, gt_category)
            finetuned_scores = self.evaluate_response_alignment(finetuned_response, gt_category)
            
            # Get best reference match for context
            relevant_sources = self.fraud_categories.get(gt_category, self.fraud_categories['general'])
            best_ref_idx = finetuned_scores.get('best_reference_idx', 0)
            best_reference = relevant_sources[best_ref_idx] if relevant_sources else ""
            
            # Create evaluation result
            result = EvaluationResult(
                question_id=question_id,
                category=category,
                question=question,
                baseline_response=baseline_response,
                finetuned_response=finetuned_response,
                baseline_best_match=baseline_scores['best_match_score'],
                baseline_avg_alignment=baseline_scores['average_alignment'],
                finetuned_best_match=finetuned_scores['best_match_score'],
                finetuned_avg_alignment=finetuned_scores['average_alignment'],
                best_match_improvement=finetuned_scores['best_match_score'] - baseline_scores['best_match_score'],
                avg_alignment_improvement=finetuned_scores['average_alignment'] - baseline_scores['average_alignment'],
                best_reference_match=best_reference[:200] + "..." if len(best_reference) > 200 else best_reference
            )
            
            results.append(result)
            
        # Calculate comprehensive statistics
        summary_stats = self._calculate_summary_statistics(results)
        
        # Generate comprehensive visualizations
        self._generate_comprehensive_visualizations(results, summary_stats)
        
        return {
            'evaluation_results': results,
            'summary_statistics': summary_stats,
            'methodology': 'Standard Cosine Similarity (Academic Approach)',
            'ground_truth_size': len(self.ground_truth_data),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_summary_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics following academic standards"""
        
        # Extract score arrays
        baseline_best = [r.baseline_best_match for r in results]
        finetuned_best = [r.finetuned_best_match for r in results]
        baseline_avg = [r.baseline_avg_alignment for r in results]
        finetuned_avg = [r.finetuned_avg_alignment for r in results]
        
        best_improvements = [r.best_match_improvement for r in results]
        avg_improvements = [r.avg_alignment_improvement for r in results]
        
        # Statistical tests
        best_t_stat, best_p_value = stats.ttest_rel(finetuned_best, baseline_best)
        avg_t_stat, avg_p_value = stats.ttest_rel(finetuned_avg, baseline_avg)
        
        # Effect sizes (Cohen's d)
        best_cohens_d = (np.mean(finetuned_best) - np.mean(baseline_best)) / np.sqrt(
            ((np.std(baseline_best)**2) + (np.std(finetuned_best)**2)) / 2)
        avg_cohens_d = (np.mean(finetuned_avg) - np.mean(baseline_avg)) / np.sqrt(
            ((np.std(baseline_avg)**2) + (np.std(finetuned_avg)**2)) / 2)
        
        return {
            'sample_size': len(results),
            'best_match_scores': {
                'baseline_mean': float(np.mean(baseline_best)),
                'baseline_std': float(np.std(baseline_best)),
                'finetuned_mean': float(np.mean(finetuned_best)),
                'finetuned_std': float(np.std(finetuned_best)),
                'mean_improvement': float(np.mean(best_improvements)),
                'improvement_std': float(np.std(best_improvements)),
                'improvement_consistency': float(np.sum(np.array(best_improvements) > 0) / len(best_improvements)),
                't_statistic': float(best_t_stat),
                'p_value': float(best_p_value),
                'cohens_d': float(best_cohens_d),
                'significant': bool(best_p_value < 0.05)
            },
            'average_alignment_scores': {
                'baseline_mean': float(np.mean(baseline_avg)),
                'baseline_std': float(np.std(baseline_avg)),
                'finetuned_mean': float(np.mean(finetuned_avg)),
                'finetuned_std': float(np.std(finetuned_avg)),
                'mean_improvement': float(np.mean(avg_improvements)),
                'improvement_std': float(np.std(avg_improvements)),
                'improvement_consistency': float(np.sum(np.array(avg_improvements) > 0) / len(avg_improvements)),
                't_statistic': float(avg_t_stat),
                'p_value': float(avg_p_value),
                'cohens_d': float(avg_cohens_d),
                'significant': bool(avg_p_value < 0.05)
            }
        }
    
    def _setup_academic_style(self):
        """Configure matplotlib for publication-quality academic figures"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'patch.linewidth': 1,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.family': 'DejaVu Sans'
        })
        
        # Set color palette for consistency
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#16537e']
        sns.set_palette(colors)
    
    def _generate_comprehensive_visualizations(self, results: List[EvaluationResult], summary_stats: Dict[str, Any]):
        """Generate comprehensive academic-quality visualizations"""
        logger.info("Generating comprehensive academic visualizations...")
        
        # Setup academic styling
        self._setup_academic_style()
        
        # Create visualization output directory
        vis_dir = Path("visualization_results")
        vis_dir.mkdir(exist_ok=True)
        
        # Generate all visualization components
        self._plot_overall_performance_comparison(results, summary_stats, vis_dir)
        self._plot_category_performance_breakdown(results, vis_dir)
        self._plot_statistical_significance_analysis(summary_stats, vis_dir)
        self._plot_improvement_distribution(results, vis_dir)
        self._plot_individual_question_ranking(results, vis_dir)
        self._plot_effect_size_visualization(summary_stats, vis_dir)
        self._plot_category_correlation_matrix(results, vis_dir)
        self._plot_confidence_intervals(summary_stats, vis_dir)
        
        logger.info(f"All visualizations saved to {vis_dir}/")
        
    def _plot_overall_performance_comparison(self, results: List[EvaluationResult], summary_stats: Dict[str, Any], output_dir: Path):
        """Generate overall performance comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # TODO: Modify later
        # fig.suptitle('Standard Cosine Similarity Evaluation - Overall Performance Analysis', 
        #             fontsize=16, fontweight='bold', y=0.95)
        
        # Extract data
        baseline_best = [r.baseline_best_match for r in results]
        finetuned_best = [r.finetuned_best_match for r in results]
        improvements = [r.best_match_improvement for r in results]
        
        # Plot 1: Box plot comparison
        data_for_box = [baseline_best, finetuned_best]
        labels = ['Baseline Model', 'Fine-tuned Model']
        bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax1.set_title('Best Match Score Distribution Comparison')
        ax1.set_ylabel('Cosine Similarity Score')
        ax1.grid(True, alpha=0.3)
        
        # Add significance annotation
        p_val = summary_stats['best_match_scores']['p_value']
        ax1.annotate(f'p = {p_val:.4f}', xy=(1.5, 0.85), xytext=(1.5, 0.9),
                    ha='center', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Scatter plot with improvement line
        question_ids = [int(r.question_id[1:]) for r in results]
        ax2.scatter(baseline_best, finetuned_best, alpha=0.6, s=60)
        
        # Add diagonal line for reference (no improvement)
        min_val, max_val = min(min(baseline_best), min(finetuned_best)), max(max(baseline_best), max(finetuned_best))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No Improvement Line')
        ax2.set_xlabel('Baseline Best Match Score')
        ax2.set_ylabel('Fine-tuned Best Match Score') 
        ax2.set_title('Score Correlation: Baseline vs Fine-tuned')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement histogram
        ax3.hist(improvements, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(improvements):.3f}')
        ax3.set_xlabel('Best Match Score Improvement')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Performance Improvements')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative improvement
        sorted_improvements = sorted(improvements, reverse=True)
        cumulative_pct = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements) * 100
        ax4.plot(sorted_improvements, cumulative_pct, 'b-', linewidth=2, marker='o', markersize=3)
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax4.set_xlabel('Best Match Score Improvement')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('Cumulative Distribution of Improvements')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'overall_performance_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_category_performance_breakdown(self, results: List[EvaluationResult], output_dir: Path):
        """Generate category-specific performance breakdown"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Category-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        # Group by category
        category_data = {}
        for r in results:
            if r.category not in category_data:
                category_data[r.category] = {
                    'baseline': [], 'finetuned': [], 'improvements': []
                }
            category_data[r.category]['baseline'].append(r.baseline_best_match)
            category_data[r.category]['finetuned'].append(r.finetuned_best_match)
            category_data[r.category]['improvements'].append(r.best_match_improvement)
        
        # Calculate category statistics
        categories = list(category_data.keys())
        baseline_means = [np.mean(category_data[cat]['baseline']) for cat in categories]
        finetuned_means = [np.mean(category_data[cat]['finetuned']) for cat in categories]
        improvement_means = [np.mean(category_data[cat]['improvements']) for cat in categories]
        improvement_stds = [np.std(category_data[cat]['improvements']) for cat in categories]
        
        # Plot 1: Category comparison bar chart
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8)
        bars2 = ax1.bar(x + width/2, finetuned_means, width, label='Fine-tuned', alpha=0.8)
        
        ax1.set_xlabel('Evaluation Categories')
        ax1.set_ylabel('Mean Best Match Score')
        ax1.set_title('Performance by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.replace(' Assessment', '').replace(' Coverage', '') for cat in categories], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Category improvement with error bars
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(categories)))
        bars = ax2.bar(x, improvement_means, yerr=improvement_stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax2.set_xlabel('Evaluation Categories')
        ax2.set_ylabel('Mean Improvement ± SD')
        ax2.set_title('Category-Specific Improvements')
        ax2.set_xticks(x)
        ax2.set_xticklabels([cat.replace(' Assessment', '').replace(' Coverage', '') for cat in categories], 
                           rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, improvement_means, improvement_stds)):
            ax2.annotate(f'{mean_val:.3f}±{std_val:.3f}', 
                        xy=(bar.get_x() + bar.get_width()/2, mean_val + std_val),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'category_performance_breakdown.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'category_performance_breakdown.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_statistical_significance_analysis(self, summary_stats: Dict[str, Any], output_dir: Path):
        """Generate statistical significance analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        best_stats = summary_stats['best_match_scores']
        avg_stats = summary_stats['average_alignment_scores']
        
        # Plot 1: Effect size visualization with interpretation bands
        metrics = ['Best Match\nScores', 'Average\nAlignment']
        effect_sizes = [best_stats['cohens_d'], avg_stats['cohens_d']]
        colors = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
        
        bars = ax1.bar(metrics, effect_sizes, color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='Small Effect (0.2)')
        ax1.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Medium Effect (0.5)')
        ax1.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Large Effect (0.8)')
        ax1.set_ylabel("Cohen's d (Effect Size)")
        ax1.set_title('Effect Size Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, es in zip(bars, effect_sizes):
            ax1.annotate(f'{es:.3f}', xy=(bar.get_x() + bar.get_width()/2, es),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                        fontweight='bold')
        
        # Plot 2: P-value visualization
        p_values = [best_stats['p_value'], avg_stats['p_value']]
        significance = ['Significant' if p < 0.05 else 'Not Significant' for p in p_values]
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        
        bars = ax2.bar(metrics, [-np.log10(p) for p in p_values], color=colors, alpha=0.8, edgecolor='black')
        ax2.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
        ax2.axhline(-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_title('Statistical Significance (p-values)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add p-value and significance labels
        for bar, p_val, sig in zip(bars, p_values, significance):
            ax2.annotate(f'p = {p_val:.4f}\n{sig}', 
                        xy=(bar.get_x() + bar.get_width()/2, -np.log10(p_val)),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                        fontweight='bold')
        
        # Plot 3: Confidence intervals
        means = [best_stats['mean_improvement'], avg_stats['mean_improvement']]
        stds = [best_stats['improvement_std'], avg_stats['improvement_std']]
        n_samples = summary_stats['sample_size']
        
        # Calculate 95% CI
        ci_95 = [1.96 * std / np.sqrt(n_samples) for std in stds]
        
        ax3.bar(metrics, means, yerr=ci_95, capsize=10, alpha=0.8, 
                color=['green' if m > 0 else 'red' for m in means], edgecolor='black')
        ax3.axhline(0, color='red', linestyle='-', alpha=0.7, label='No Improvement')
        ax3.set_ylabel('Mean Improvement')
        ax3.set_title('95% Confidence Intervals for Improvements')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, ci) in enumerate(zip(means, ci_95)):
            ax3.annotate(f'{mean:.3f}±{ci:.3f}', 
                        xy=(i, mean + ci), xytext=(0, 3), 
                        textcoords="offset points", ha='center', va='bottom', 
                        fontweight='bold')
        
        # Plot 4: Improvement consistency
        consistencies = [best_stats['improvement_consistency']*100, avg_stats['improvement_consistency']*100]
        colors = ['green' if c > 60 else 'orange' if c > 50 else 'red' for c in consistencies]
        
        bars = ax4.bar(metrics, consistencies, color=colors, alpha=0.8, edgecolor='black')
        ax4.axhline(50, color='red', linestyle='--', alpha=0.7, label='Random Chance (50%)')
        ax4.axhline(60, color='orange', linestyle='--', alpha=0.7, label='Good Consistency (60%)')
        ax4.set_ylabel('Improvement Consistency (%)')
        ax4.set_title('Percentage of Questions with Improvement')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for bar, consistency in zip(bars, consistencies):
            ax4.annotate(f'{consistency:.1f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, consistency),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                        fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_significance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'statistical_significance_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_improvement_distribution(self, results: List[EvaluationResult], output_dir: Path):
        """Generate improvement distribution analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # fig.suptitle('Improvement Distribution Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        improvements = [r.best_match_improvement for r in results]
        question_ids = [int(r.question_id[1:]) for r in results]
        
        # Plot 1: Distribution histogram with statistics
        ax1.hist(improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add normal distribution overlay
        mu, sigma = np.mean(improvements), np.std(improvements)
        x = np.linspace(min(improvements), max(improvements), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax1.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mu:.3f}')
        ax1.axvline(0, color='black', linestyle=':', alpha=0.7, label='No Improvement')
        ax1.set_xlabel('Best Match Score Improvement')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Improvements with Normal Overlay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot for normality
        from scipy.stats import probplot
        probplot(improvements, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Normality Check')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement by question number
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax3.scatter(question_ids, improvements, c=colors, alpha=0.7, s=60)
        ax3.axhline(0, color='black', linestyle='--', alpha=0.7, label='No Improvement')
        ax3.axhline(np.mean(improvements), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(improvements):.3f}')
        ax3.set_xlabel('Question Number')
        ax3.set_ylabel('Best Match Score Improvement')
        ax3.set_title('Improvement by Question')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot with outlier analysis
        bp = ax4.boxplot(improvements, patch_artist=True, 
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax4.set_ylabel('Best Match Score Improvement')
        ax4.set_title('Box Plot with Outlier Detection')
        ax4.grid(True, alpha=0.3)
        
        # Add statistical annotations
        q1, median, q3 = np.percentile(improvements, [25, 50, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [x for x in improvements if x < lower_fence or x > upper_fence]
        
        stats_text = f'Median: {median:.3f}\nQ1: {q1:.3f}\nQ3: {q3:.3f}\nIQR: {iqr:.3f}\nOutliers: {len(outliers)}'
        ax4.text(1.15, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'improvement_distribution_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_individual_question_ranking(self, results: List[EvaluationResult], output_dir: Path):
        """Generate individual question performance ranking"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Individual Question Performance Ranking', fontsize=16, fontweight='bold')
        
        # Sort by improvement
        sorted_results = sorted(results, key=lambda x: x.best_match_improvement, reverse=True)
        
        # Top 15 and bottom 10 for readability
        top_results = sorted_results[:15]
        bottom_results = sorted_results[-10:]
        
        # Plot 1: Top performers
        top_improvements = [r.best_match_improvement for r in top_results]
        top_labels = [f"{r.question_id}\n{r.category.replace(' Assessment', '').replace(' Coverage', '')}" for r in top_results]
        
        y_pos = np.arange(len(top_results))
        colors = ['darkgreen' if imp > 0.05 else 'green' if imp > 0 else 'red' for imp in top_improvements]
        
        bars = ax1.barh(y_pos, top_improvements, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_labels, fontsize=9)
        ax1.set_xlabel('Best Match Score Improvement')
        ax1.set_title('Top 15 Performing Questions')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, improvement in zip(bars, top_improvements):
            width = bar.get_width()
            ax1.annotate(f'{improvement:.3f}', 
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3 if width >= 0 else -3, 0), textcoords="offset points",
                        ha='left' if width >= 0 else 'right', va='center', fontsize=8, fontweight='bold')
        
        # Plot 2: Bottom performers (areas of concern)
        bottom_improvements = [r.best_match_improvement for r in bottom_results]
        bottom_labels = [f"{r.question_id}\n{r.category.replace(' Assessment', '').replace(' Coverage', '')}" for r in bottom_results]
        
        y_pos = np.arange(len(bottom_results))
        colors = ['darkred' if imp < -0.02 else 'red' if imp < 0 else 'orange' for imp in bottom_improvements]
        
        bars = ax2.barh(y_pos, bottom_improvements, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(bottom_labels, fontsize=9)
        ax2.set_xlabel('Best Match Score Improvement')
        ax2.set_title('Bottom 10 Performing Questions (Areas of Concern)')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, improvement in zip(bars, bottom_improvements):
            width = bar.get_width()
            ax2.annotate(f'{improvement:.3f}', 
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(3 if width >= 0 else -3, 0), textcoords="offset points",
                        ha='left' if width >= 0 else 'right', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'individual_question_ranking.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'individual_question_ranking.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_effect_size_visualization(self, summary_stats: Dict[str, Any], output_dir: Path):
        """Generate effect size interpretation visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Effect Size Analysis and Interpretation', fontsize=16, fontweight='bold')
        
        best_stats = summary_stats['best_match_scores']
        avg_stats = summary_stats['average_alignment_scores']
        
        # Plot 1: Effect size with interpretation bands
        effect_sizes = [best_stats['cohens_d'], avg_stats['cohens_d']]
        metrics = ['Best Match Scores', 'Average Alignment']
        
        # Create interpretation bands
        ax1.axhspan(-0.2, 0.2, alpha=0.2, color='red', label='Negligible Effect (|d| < 0.2)')
        ax1.axhspan(0.2, 0.5, alpha=0.2, color='orange', label='Small Effect (0.2 ≤ |d| < 0.5)')
        ax1.axhspan(0.5, 0.8, alpha=0.2, color='yellow', label='Medium Effect (0.5 ≤ |d| < 0.8)')
        ax1.axhspan(0.8, 1.5, alpha=0.2, color='green', label='Large Effect (|d| ≥ 0.8)')
        
        # Plot effect sizes
        colors = ['green' if abs(es) >= 0.8 else 'yellow' if abs(es) >= 0.5 else 'orange' if abs(es) >= 0.2 else 'red' 
                 for es in effect_sizes]
        bars = ax1.bar(metrics, effect_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax1.set_ylabel("Cohen's d (Effect Size)")
        ax1.set_title('Effect Size Magnitude with Interpretation Bands')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and interpretations
        interpretations = []
        for es in effect_sizes:
            if abs(es) >= 0.8:
                interpretations.append('Large Effect')
            elif abs(es) >= 0.5:
                interpretations.append('Medium Effect')
            elif abs(es) >= 0.2:
                interpretations.append('Small Effect')
            else:
                interpretations.append('Negligible Effect')
        
        for bar, es, interp in zip(bars, effect_sizes, interpretations):
            ax1.annotate(f'{es:.3f}\n({interp})', 
                        xy=(bar.get_x() + bar.get_width()/2, es),
                        xytext=(0, 10 if es >= 0 else -25), textcoords="offset points",
                        ha='center', va='bottom' if es >= 0 else 'top', 
                        fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot 2: Practical significance analysis
        p_values = [best_stats['p_value'], avg_stats['p_value']]
        effect_sizes_abs = [abs(es) for es in effect_sizes]
        
        # Create quadrants for interpretation
        ax2.axhspan(0, 0.2, alpha=0.1, color='gray', label='Small Effect Size')
        ax2.axhspan(0.2, 0.5, alpha=0.1, color='orange', label='Medium Effect Size') 
        ax2.axhspan(0.5, 1.0, alpha=0.1, color='green', label='Large Effect Size')
        ax2.axvline(0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
        ax2.axvline(0.01, color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
        
        # Plot points
        colors = ['green' if p < 0.05 and abs_es >= 0.2 else 'orange' if p < 0.05 else 'red' 
                 for p, abs_es in zip(p_values, effect_sizes_abs)]
        
        scatter = ax2.scatter(p_values, effect_sizes_abs, c=colors, s=200, alpha=0.8, 
                            edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, (p, es_abs, metric) in enumerate(zip(p_values, effect_sizes_abs, metrics)):
            ax2.annotate(metric, xy=(p, es_abs), xytext=(10, 10), 
                        textcoords="offset points", ha='left', va='bottom',
                        fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('p-value (log scale)')
        ax2.set_ylabel('|Effect Size| (Cohen\'s d)')
        ax2.set_title('Statistical vs Practical Significance')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'effect_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'effect_size_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_category_correlation_matrix(self, results: List[EvaluationResult], output_dir: Path):
        """Generate category correlation analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Category Performance Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data for correlation analysis
        categories = list(set(r.category for r in results))
        category_short = [cat.replace(' Assessment', '').replace(' Coverage', '') for cat in categories]
        
        # Create correlation matrices
        baseline_matrix = np.zeros((len(categories), len(categories)))
        improvement_matrix = np.zeros((len(categories), len(categories)))
        
        # Group results by category
        category_results = {cat: [] for cat in categories}
        for r in results:
            category_results[r.category].append(r)
        
        # Calculate correlations between categories
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i == j:
                    baseline_matrix[i, j] = 1.0
                    improvement_matrix[i, j] = 1.0
                else:
                    # Get all baseline scores for each category
                    cat1_baseline = [r.baseline_best_match for r in category_results[cat1]]
                    cat2_baseline = [r.baseline_best_match for r in category_results[cat2]]
                    
                    cat1_improvement = [r.best_match_improvement for r in category_results[cat1]]
                    cat2_improvement = [r.best_match_improvement for r in category_results[cat2]]
                    
                    # Calculate correlation if we have enough data
                    if len(cat1_baseline) > 1 and len(cat2_baseline) > 1:
                        # For different lengths, we'll use the mean values
                        baseline_corr = np.corrcoef([np.mean(cat1_baseline)], [np.mean(cat2_baseline)])[0, 1]
                        improvement_corr = np.corrcoef([np.mean(cat1_improvement)], [np.mean(cat2_improvement)])[0, 1]
                        
                        baseline_matrix[i, j] = baseline_corr if not np.isnan(baseline_corr) else 0
                        improvement_matrix[i, j] = improvement_corr if not np.isnan(improvement_corr) else 0
                    else:
                        baseline_matrix[i, j] = 0
                        improvement_matrix[i, j] = 0
        
        # Plot 1: Baseline performance correlation
        im1 = ax1.imshow(baseline_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(category_short)))
        ax1.set_yticks(range(len(category_short)))
        ax1.set_xticklabels(category_short, rotation=45, ha='right')
        ax1.set_yticklabels(category_short)
        ax1.set_title('Baseline Performance Correlation')
        
        # Add correlation values to cells
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = ax1.text(j, i, f'{baseline_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(baseline_matrix[i, j]) < 0.5 else "white",
                              fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Plot 2: Improvement correlation
        im2 = ax2.imshow(improvement_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(category_short)))
        ax2.set_yticks(range(len(category_short)))
        ax2.set_xticklabels(category_short, rotation=45, ha='right')
        ax2.set_yticklabels(category_short)
        ax2.set_title('Improvement Correlation')
        
        # Add correlation values to cells
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = ax2.text(j, i, f'{improvement_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(improvement_matrix[i, j]) < 0.5 else "white",
                              fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'category_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'category_correlation_analysis.pdf', bbox_inches='tight')
        plt.close()
        
    def _plot_confidence_intervals(self, summary_stats: Dict[str, Any], output_dir: Path):
        """Generate confidence interval analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # fig.suptitle('Confidence Interval Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        best_stats = summary_stats['best_match_scores']
        avg_stats = summary_stats['average_alignment_scores']
        n_samples = summary_stats['sample_size']
        
        # Calculate different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        z_scores = [1.645, 1.96, 2.576]  # For 90%, 95%, 99%
        
        # Plot 1: Best match confidence intervals
        means = [best_stats['baseline_mean'], best_stats['finetuned_mean']]
        stds = [best_stats['baseline_std'], best_stats['finetuned_std']]
        labels = ['Baseline', 'Fine-tuned']
        
        x_pos = np.arange(len(labels))
        colors = ['lightcoral', 'lightblue']
        
        for i, (mean, std, color) in enumerate(zip(means, stds, colors)):
            # Plot point estimate
            ax1.scatter([i], [mean], color=color, s=100, zorder=3, edgecolor='black', linewidth=2)
            
            # Plot confidence intervals
            for j, (conf_level, z_score) in enumerate(zip(confidence_levels, z_scores)):
                margin_error = z_score * std / np.sqrt(n_samples)
                ci_lower = mean - margin_error
                ci_upper = mean + margin_error
                
                # Different line styles for different confidence levels
                line_styles = ['-', '--', ':']
                alphas = [0.8, 0.6, 0.4]
                
                ax1.plot([i, i], [ci_lower, ci_upper], color=color, 
                        linestyle=line_styles[j], linewidth=3, alpha=alphas[j],
                        label=f'{conf_level*100:.0f}% CI' if i == 0 else "")
                
                # Add error bars caps
                cap_size = 0.05
                ax1.plot([i-cap_size, i+cap_size], [ci_lower, ci_lower], color=color, 
                        linestyle=line_styles[j], linewidth=2, alpha=alphas[j])
                ax1.plot([i-cap_size, i+cap_size], [ci_upper, ci_upper], color=color, 
                        linestyle=line_styles[j], linewidth=2, alpha=alphas[j])
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Best Match Score')
        ax1.set_title('Best Match Score Confidence Intervals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement confidence intervals
        improvement_mean = best_stats['mean_improvement']
        improvement_std = best_stats['improvement_std']
        
        # Different confidence intervals for the improvement
        ci_data = []
        for conf_level, z_score in zip(confidence_levels, z_scores):
            margin_error = z_score * improvement_std / np.sqrt(n_samples)
            ci_lower = improvement_mean - margin_error
            ci_upper = improvement_mean + margin_error
            ci_data.append((conf_level, ci_lower, ci_upper, margin_error))
        
        y_pos = np.arange(len(confidence_levels))
        colors = ['green', 'orange', 'red']
        
        for i, (conf_level, ci_lower, ci_upper, margin_error) in enumerate(ci_data):
            # Plot the interval
            ax2.barh(i, ci_upper - ci_lower, left=ci_lower, 
                    color=colors[i], alpha=0.6, edgecolor='black',
                    label=f'{conf_level*100:.0f}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
            
            # Mark the point estimate
            ax2.scatter([improvement_mean], [i], color='red', s=100, zorder=3, 
                       edgecolor='black', linewidth=2)
        
        ax2.axvline(0, color='black', linestyle='--', alpha=0.7, label='No Improvement')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'{cl*100:.0f}%' for cl in confidence_levels])
        ax2.set_xlabel('Best Match Score Improvement')
        ax2.set_ylabel('Confidence Level')
        ax2.set_title('Improvement Confidence Intervals')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Sample size impact on confidence intervals
        sample_sizes = np.arange(10, 101, 10)
        margins_95 = [1.96 * improvement_std / np.sqrt(n) for n in sample_sizes]
        margins_99 = [2.576 * improvement_std / np.sqrt(n) for n in sample_sizes]
        
        ax3.plot(sample_sizes, margins_95, 'b-', linewidth=2, label='95% CI Margin')
        ax3.plot(sample_sizes, margins_99, 'r-', linewidth=2, label='99% CI Margin')
        ax3.axvline(n_samples, color='green', linestyle='--', linewidth=2, 
                   label=f'Current Sample Size (n={n_samples})')
        ax3.scatter([n_samples], [1.96 * improvement_std / np.sqrt(n_samples)], 
                   color='blue', s=100, zorder=3)
        ax3.scatter([n_samples], [2.576 * improvement_std / np.sqrt(n_samples)], 
                   color='red', s=100, zorder=3)
        
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Margin of Error')
        ax3.set_title('Impact of Sample Size on Confidence Intervals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Power analysis visualization
        # Effect sizes to test
        effect_sizes_test = np.linspace(0, 0.6, 50)
        alpha = 0.05
        
        # Calculate power for different effect sizes (simplified)
        powers = []
        for es in effect_sizes_test:
            # Simplified power calculation (normally would use more sophisticated methods)
            z_alpha = 1.96  # For alpha = 0.05
            z_beta = es * np.sqrt(n_samples / 2)  # Simplified calculation
            power = 1 - stats.norm.cdf(z_alpha - z_beta)
            powers.append(max(0, min(1, power)))  # Clamp between 0 and 1
        
        ax4.plot(effect_sizes_test, powers, 'b-', linewidth=2, label='Power Curve')
        ax4.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')
        ax4.axvline(best_stats['cohens_d'], color='green', linestyle='--', linewidth=2,
                   label=f'Observed Effect Size ({best_stats["cohens_d"]:.3f})')
        
        # Mark the observed power
        observed_power_idx = np.argmin(np.abs(effect_sizes_test - best_stats['cohens_d']))
        observed_power = powers[observed_power_idx]
        ax4.scatter([best_stats['cohens_d']], [observed_power], 
                   color='green', s=100, zorder=3, edgecolor='black', linewidth=2)
        ax4.annotate(f'Observed Power ≈ {observed_power:.2f}', 
                    xy=(best_stats['cohens_d'], observed_power),
                    xytext=(10, 10), textcoords="offset points",
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax4.set_xlabel('Effect Size (Cohen\'s d)')
        ax4.set_ylabel('Statistical Power')
        ax4.set_title('Power Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_interval_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'confidence_interval_analysis.pdf', bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    
    # Initialize evaluator
    evaluator = StandardCosineEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Create output directory
    output_dir = Path("standard_cosine_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "standard_cosine_results.json", 'w') as f:
        # Convert EvaluationResult objects to dictionaries for JSON serialization
        json_results = {
            'evaluation_results': [
                {
                    'question_id': r.question_id,
                    'category': r.category, 
                    'question': r.question,
                    'baseline_response': r.baseline_response,
                    'finetuned_response': r.finetuned_response,
                    'baseline_best_match': r.baseline_best_match,
                    'baseline_avg_alignment': r.baseline_avg_alignment,
                    'finetuned_best_match': r.finetuned_best_match,
                    'finetuned_avg_alignment': r.finetuned_avg_alignment,
                    'best_match_improvement': r.best_match_improvement,
                    'avg_alignment_improvement': r.avg_alignment_improvement,
                    'best_reference_match': r.best_reference_match
                } for r in results['evaluation_results']
            ],
            'summary_statistics': results['summary_statistics'],
            'methodology': results['methodology'],
            'ground_truth_size': results['ground_truth_size'],
            'embedding_model': results['embedding_model'],
            'evaluation_timestamp': results['evaluation_timestamp']
        }
        json.dump(json_results, f, indent=2)
    
    # Print summary
    stats = results['summary_statistics']
    print("\n" + "="*80)
    print("STANDARD COSINE SIMILARITY EVALUATION RESULTS")
    print("="*80)
    print(f"Methodology: {results['methodology']}")
    print(f"Ground Truth Sources: {results['ground_truth_size']} authoritative Q&A pairs")
    print(f"Embedding Model: {results['embedding_model']}")
    print(f"Sample Size: {stats['sample_size']}")
    
    print(f"\nBEST MATCH SCORES:")
    best_stats = stats['best_match_scores']
    print(f"  Baseline Mean: {best_stats['baseline_mean']:.3f} ± {best_stats['baseline_std']:.3f}")
    print(f"  Fine-tuned Mean: {best_stats['finetuned_mean']:.3f} ± {best_stats['finetuned_std']:.3f}")
    print(f"  Mean Improvement: {best_stats['mean_improvement']:.3f} ± {best_stats['improvement_std']:.3f}")
    print(f"  Improvement Consistency: {best_stats['improvement_consistency']*100:.1f}%")
    print(f"  Statistical Significance: {'Yes' if best_stats['significant'] else 'No'} (p = {best_stats['p_value']:.4f})")
    print(f"  Effect Size: {best_stats['cohens_d']:.3f} (Cohen's d)")
    
    print(f"\nAVERAGE ALIGNMENT SCORES:")
    avg_stats = stats['average_alignment_scores']
    print(f"  Baseline Mean: {avg_stats['baseline_mean']:.3f} ± {avg_stats['baseline_std']:.3f}")
    print(f"  Fine-tuned Mean: {avg_stats['finetuned_mean']:.3f} ± {avg_stats['finetuned_std']:.3f}")
    print(f"  Mean Improvement: {avg_stats['mean_improvement']:.3f} ± {avg_stats['improvement_std']:.3f}")
    print(f"  Improvement Consistency: {avg_stats['improvement_consistency']*100:.1f}%")
    print(f"  Statistical Significance: {'Yes' if avg_stats['significant'] else 'No'} (p = {avg_stats['p_value']:.4f})")
    print(f"  Effect Size: {avg_stats['cohens_d']:.3f} (Cohen's d)")
    
    print(f"\nResults saved to: {output_dir}/")
    logger.info("Standard Cosine Similarity evaluation completed!")

if __name__ == "__main__":
    main()