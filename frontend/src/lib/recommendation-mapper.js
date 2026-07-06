// src/lib/recommendation-mapper.js

/**
 * Dynamically generates a list of actionable recommendations based on the 
 * domain scores and severity tiers returned by the backend.
 *
 * @param {Object} domainScores - The domain_scores object from compiled_profile.json
 * @returns {Object} An object mapping domain keys to arrays of recommendation strings.
 */
export function generateDynamicRecommendations(domainScores) {
  if (!domainScores) return {};

  const recommendations = {};

  // Domain 2: Self-Esteem
  const selfEsteem = domainScores.domain_2_self_esteem;
  if (selfEsteem) {
    const severity = selfEsteem.severity_tier?.toLowerCase();
    const recs = [];
    if (severity === 'high' || severity === 'severe') {
      recs.push("Consider cognitive behavioral strategies to challenge persistent negative self-evaluations.");
      recs.push("Engage in structured self-compassion exercises daily.");
    } else if (severity === 'moderate') {
      recs.push("Practice acknowledging personal strengths and achievements.");
      recs.push("Monitor and reframe harsh internal dialogue.");
    } else {
      recs.push("Maintain current positive self-reflection practices.");
      recs.push("Continue leveraging your strong sense of self-efficacy in daily challenges.");
    }
    recommendations['self_esteem'] = { severity, recs };
  }

  // Domain 3: Mood & Sleep
  const moodSleep = domainScores.domain_3_mood_sleep;
  if (moodSleep) {
    const severity = moodSleep.severity_tier?.toLowerCase();
    const sleepHours = moodSleep.placement?.calculated_sleep_duration_hours || 0;
    const recs = [];
    
    if (severity === 'severe' || severity === 'high') {
      recs.push("Strongly consider consulting a mental health professional for a comprehensive mood evaluation.");
    } else if (severity === 'moderate') {
      recs.push("Implement daily mindfulness or mood-tracking to identify emotional triggers.");
    } else {
      recs.push("Continue maintaining your current mood management and coping strategies.");
    }

    if (sleepHours < 6) {
      recs.push("Prioritize sleep hygiene to increase duration to the recommended 7-9 hours.");
    } else if (sleepHours > 10) {
      recs.push("Monitor hypersomnia patterns and consider setting consistent wake times.");
    } else {
      recs.push("Maintain your current healthy sleep schedule.");
    }
    recommendations['mood_sleep'] = { severity, recs };
  }

  // Domain 4: Digital & Social
  const digital = domainScores.domain_4_digital_and_social || domainScores.domain_4_multitask;
  if (digital) {
    const severity = digital.severity_tier?.toLowerCase();
    const recs = [];
    if (severity === 'severe' || severity === 'high') {
      recs.push("Actively reduce non-essential screen time to mitigate cross-impact depression risks.");
      recs.push("Schedule regular, mandatory offline periods and prioritize in-person social interactions.");
    } else if (severity === 'moderate') {
      recs.push("Set structured boundaries for digital consumption, especially before bedtime.");
      recs.push("Balance virtual relationships with physical community engagement.");
    } else {
      recs.push("Continue maintaining a healthy boundary between digital use and offline life.");
    }
    recommendations['digital_and_social'] = { severity, recs };
  }

  // Domain 5: Occupational Burnout
  const burnout = domainScores.domain_5_occupational_burnout;
  if (burnout) {
    const severity = burnout.severity_tier?.toLowerCase();
    const recs = [];
    if (severity === 'severe' || severity === 'high') {
      recs.push("Immediate intervention recommended: Discuss workload recalibration with supervisors or HR.");
      recs.push("Establish strict boundaries between professional output hours and personal recovery time.");
    } else if (severity === 'moderate') {
      recs.push("Identify primary workplace stressors and seek additional social or structural support.");
      recs.push("Incorporate micro-breaks and decompression routines into your daily schedule.");
    } else {
      recs.push("Your occupational engagement is well-controlled. Continue current work-life balance strategies.");
    }
    recommendations['occupational_burnout'] = { severity, recs };
  }

  // Domain 6: Clinical Severity
  const clinical = domainScores.domain_6_severe_clinical;
  if (clinical) {
    const severity = clinical.severity_tier?.toLowerCase();
    const flag = clinical.placement?.anomaly_review_flag;
    const recs = [];
    
    if (flag) {
      recs.push("URGENT: Atypical response patterns detected. A comprehensive differential evaluation by a clinical professional is strongly recommended.");
    } else if (severity === 'severe' || severity === 'high') {
      recs.push("Clinical screening indicates elevated risk markers. Professional consultation is advised to evaluate these signals.");
    } else {
      recs.push("No severe clinical anomalies detected in the current screening baseline.");
    }
    recommendations['severe_clinical'] = { severity, recs };
  }

  return recommendations;
}
