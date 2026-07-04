// src/pages/Assessment.jsx (Updated)
import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { ArrowLeft, ArrowRight, Brain, CheckCircle2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";
import QuestionCard from "../components/assessment/QuestionCard";
import ProgressBar from "../components/assessment/ProgressBar";
import { QUESTIONNAIRE_SECTIONS } from "../lib/questionnaire-data";
import { mindsightAPI } from "../lib/apiClient"; // Import API client

export default function Assessment() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState(null);
  const [responses, setResponses] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showTransition, setShowTransition] = useState(false);
  const [error, setError] = useState(null);
  const [assessmentResult, setAssessmentResult] = useState(null);

  const handleNextSection = () => {
    setShowTransition(true);
  };

  const proceedToNext = () => {
    setStep(prev => prev + 1);
    setShowTransition(false);
  };

  const totalQuestions = QUESTIONNAIRE_SECTIONS.reduce((sum, s) => sum + s.questions.length, 0);
  const answeredCount = Object.keys(responses).length;

  const currentSection = step > 0 && step <= QUESTIONNAIRE_SECTIONS.length
    ? QUESTIONNAIRE_SECTIONS[step - 1]
    : null;

  const sectionAnswered = currentSection
    ? currentSection.questions.filter(q => {
        if (q.inputType === "time") {
          return responses[q.id] && responses[q.id].trim() !== "";
        }
        if (q.inputType === "number") {
          return responses[q.id] !== undefined && responses[q.id] !== "" && !isNaN(responses[q.id]);
        }
        if (q.inputType === "radio") {
          return responses[q.id] !== undefined && responses[q.id] !== null;
        }
        return responses[q.id] !== undefined;
      }).length
    : 0;

  const canProceed = useMemo(() => {
    if (step === 0) {
      const isValidAge = age && parseInt(age) >= 13 && parseInt(age) <= 100;
      const isValidGender = gender !== null;
      return isValidAge && isValidGender;
    }
    if (currentSection) {
      const allAnswered = currentSection.questions.every(q => {
        if (q.inputType === "time") {
          return responses[q.id] && responses[q.id].trim() !== "";
        }
        if (q.inputType === "number") {
          const value = responses[q.id];
          if (q.id === "work_hours_per_week") {
            return value !== undefined && value !== "" && !isNaN(value) && parseFloat(value) >= 0 && parseFloat(value) <= 168;
          }
          if (q.id === "meetings_per_day") {
            return value !== undefined && value !== "" && !isNaN(value) && parseFloat(value) >= 0 && parseFloat(value) <= 50;
          }
          return value !== undefined && value !== "" && !isNaN(value);
        }
        if (q.inputType === "radio") {
          return responses[q.id] !== undefined && responses[q.id] !== null;
        }
        return responses[q.id] !== undefined;
      });
      return allAnswered;
    }
    return false;
  }, [step, age, gender, currentSection, responses]);

  const handleResponse = (questionId, value) => {
    console.log('📝 Response:', questionId, value);
    setResponses(prev => ({ ...prev, [questionId]: value }));
  };

  const handleTextInputChange = (questionId, value, type) => {
    if (type === "number") {
      if (value === "" || !isNaN(value)) {
        setResponses(prev => ({ ...prev, [questionId]: value }));
      }
    } else {
      setResponses(prev => ({ ...prev, [questionId]: value }));
    }
  };

  const handleSubmit = async () => {
    setError(null);
    setIsSubmitting(true);
    
    try {
      // Create the complete data object
      const allData = {
        age: parseInt(age),
        gender: gender,
        ...responses
      };
      
      console.log('📤 Submitting assessment data:', allData);
      
      // Send to backend API
      const response = await mindsightAPI.submitAssessment(allData);
      
      console.log('✅ Assessment submitted successfully:', response.data);
      
      // Store the result
      setAssessmentResult(response.data);
      
      // Short delay to show success animation
      setTimeout(() => {
        setIsSubmitting(false);
        // Navigate to dashboard with the result
        navigate('/dashboard', { 
          state: { 
            assessmentResult: response.data,
            timestamp: response.data.timestamp 
          } 
        });
      }, 1500);
      
    } catch (error) {
      console.error('❌ Submission error:', error);
      setError(error.response?.data?.message || error.message || 'Failed to submit assessment');
      setIsSubmitting(false);
    }
  };

  // Render question function (same as before)
  const renderQuestion = (q, index) => {
    // ... keep your existing renderQuestion function
    const cardClass = "card-surface-hover p-5 sm:p-6 mb-4";

    if (q.inputType === "radio") {
      return (
        <motion.div
          key={q.id}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className={cardClass}
        >
          <p className="text-sm font-medium leading-relaxed mb-4">
            <span className="text-primary font-semibold mr-2">Q{index + 1}.</span>
            {q.text}
          </p>
          <div className="flex flex-wrap gap-4">
            {q.options.map((option) => (
              <label key={option.value} className="flex items-center gap-2 cursor-pointer group">
                <div className="relative flex items-center justify-center w-5 h-5">
                  <input
                    type="radio"
                    name={q.id}
                    value={option.value}
                    checked={responses[q.id] === option.value}
                    onChange={() => handleResponse(q.id, option.value)}
                    className="peer sr-only"
                  />
                  <div className="w-4 h-4 rounded-full border border-border group-hover:border-primary peer-checked:border-primary peer-checked:border-4 peer-focus-visible:ring-2 peer-focus-visible:ring-primary peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background transition-all" />
                </div>
                <span className="text-sm text-foreground">{option.label}</span>
              </label>
            ))}
          </div>
        </motion.div>
      );
    }

    if (q.inputType === "time") {
      return (
        <motion.div
          key={q.id}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className={cardClass}
        >
          <p className="text-sm font-medium leading-relaxed mb-4">
            <span className="text-primary font-semibold mr-2">Q{index + 1}.</span>
            {q.text}
          </p>
          <div className="flex items-center gap-3">
            <Input
              type="time"
              id={q.id}
              value={responses[q.id] || ""}
              onChange={(e) => handleTextInputChange(q.id, e.target.value, "time")}
              className="max-w-[180px] focus-visible:ring-primary focus-visible:ring-offset-0 bg-background"
              placeholder={q.placeholder || "HH:MM"}
            />
          </div>
        </motion.div>
      );
    }

    if (q.inputType === "number") {
      return (
        <motion.div
          key={q.id}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className={cardClass}
        >
          <p className="text-sm font-medium leading-relaxed mb-4">
            <span className="text-primary font-semibold mr-2">Q{index + 1}.</span>
            {q.text}
          </p>
          <div className="flex items-center gap-3">
            <Input
              type="number"
              id={q.id}
              value={responses[q.id] || ""}
              onChange={(e) => handleTextInputChange(q.id, e.target.value, "number")}
              className="max-w-[200px] focus-visible:ring-primary focus-visible:ring-offset-0 bg-background"
              placeholder={q.placeholder || "Enter value"}
              min={q.min}
              max={q.max}
              step="any"
            />
          </div>
        </motion.div>
      );
    }

    if (q.inputType === "scale") {
      return (
        <motion.div
          key={q.id}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className={cardClass}
        >
          <p className="text-sm font-medium leading-relaxed mb-4">
            <span className="text-primary font-semibold mr-2">Q{index + 1}.</span>
            {q.text}
          </p>
          <div className="flex flex-wrap gap-2">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((val) => (
              <button
                key={val}
                onClick={() => handleResponse(q.id, val)}
                className={`w-10 h-10 rounded-full border transition-all text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
                  responses[q.id] === val
                    ? "border-primary bg-primary text-primary-foreground shadow-md shadow-primary/20"
                    : "border-border hover:border-primary/50 hover:bg-primary/5 bg-background text-muted-foreground"
                }`}
              >
                {val}
              </button>
            ))}
          </div>
        </motion.div>
      );
    }

    return (
      <QuestionCard
        key={q.id}
        question={q}
        scale={currentSection.scale}
        scaleValues={currentSection.scaleValues}
        value={responses[q.id]}
        onChange={handleResponse}
        index={index}
      />
    );
  };

  const bgColors = [
    "rgba(0,0,0,0)",
    "rgba(99, 102, 241, 0.15)",
    "rgba(14, 165, 233, 0.15)",
    "rgba(16, 185, 129, 0.15)",
    "rgba(168, 85, 247, 0.15)",
    "rgba(245, 158, 11, 0.15)",
    "rgba(244, 63, 94, 0.15)",
  ];

  return (
    <div className="flex min-h-screen flex-col relative overflow-hidden">
      <motion.div
        className="fixed left-1/2 top-0 -translate-x-1/2 w-[100vw] h-[60vh] rounded-[100%] pointer-events-none -z-10 blur-[120px] opacity-70"
        animate={{ backgroundColor: bgColors[step] || "rgba(0,0,0,0)" }}
        transition={{ duration: 1.5, ease: "easeInOut" }}
      />
      
      <Navbar />
      <main className="page-shell flex-1">
        <div className="page-container-narrow">

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 flex items-start gap-3"
          >
            <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-semibold">Submission Error</p>
              <p className="text-sm">{error}</p>
            </div>
          </motion.div>
        )}

        {/* Overall progress */}
        {step > 0 && step <= QUESTIONNAIRE_SECTIONS.length && (
          <ProgressBar
            current={answeredCount}
            total={totalQuestions}
            sectionName={`Section ${step} of ${QUESTIONNAIRE_SECTIONS.length}`}
          />
        )}

        <AnimatePresence mode="wait">
          {/* Submission Animation */}
          {isSubmitting && (
            <motion.div
              key="submission-transition"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="card-surface p-10 text-center space-y-6 max-w-md mx-auto mt-10"
            >
              <div className="flex justify-center mb-6">
                <div className="relative">
                  <motion.div 
                    animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0.8, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                    className="absolute inset-0 bg-primary/30 rounded-full blur-xl" 
                  />
                  <div className="w-24 h-24 bg-card border border-primary/30 rounded-2xl flex items-center justify-center relative shadow-lg">
                    <Brain className="w-12 h-12 text-primary animate-pulse" />
                  </div>
                </div>
              </div>
              <h3 className="text-2xl font-display font-bold text-foreground tracking-tight">Synthesizing Profile...</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                MINDSIGHT is orchestrating multi-domain models and extracting diagnostic vectors.
              </p>
              <div className="w-full bg-secondary rounded-full h-1.5 mt-8 overflow-hidden">
                <motion.div
                  className="bg-primary h-full rounded-full"
                  initial={{ width: "0%" }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 2.8, ease: "circOut" }}
                />
              </div>
            </motion.div>
          )}

          {/* Step 0: Introduction */}
          {!isSubmitting && step === 0 && (
            <motion.div
              key="intro"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
            >
              <Card className="card-surface">
                <CardHeader className="text-center pb-4 pt-8">
                  <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-5 shadow-sm border border-primary/20">
                    <Brain className="w-8 h-8 text-primary" />
                  </div>
                  <CardTitle className="text-3xl font-display text-foreground">Mental Health Assessment</CardTitle>
                  <p className="text-muted-foreground mt-3 max-w-md mx-auto leading-relaxed">
                    This assessment covers 12 psychological dimensions using clinically validated scales. 
                    It takes approximately 10-15 minutes to complete.
                  </p>
                </CardHeader>
                <CardContent className="space-y-8 px-4 sm:px-8 pb-8">

                  <div className="space-y-6">
                    <div className="space-y-3">
                      <Label htmlFor="age" className="text-sm font-semibold text-foreground">Your Age</Label>
                      <Input
                        id="age"
                        type="number"
                        placeholder="Enter your age (13-100)"
                        value={age}
                        onChange={e => setAge(e.target.value)}
                        min={13}
                        max={100}
                        className="max-w-[240px] focus-visible:ring-primary focus-visible:ring-offset-0 bg-background h-11"
                      />
                    </div>

                    <div className="space-y-3">
                      <Label className="text-sm font-semibold text-foreground">Biological Sex Assigned at Birth</Label>
                      <div className="flex flex-wrap gap-4">
                        <label className="flex items-center gap-2 cursor-pointer group">
                          <div className="relative flex items-center justify-center w-5 h-5">
                            <input
                              type="radio"
                              name="gender"
                              value="0"
                              checked={gender === 0}
                              onChange={() => setGender(0)}
                              className="peer sr-only"
                            />
                            <div className="w-4 h-4 rounded-full border border-border group-hover:border-primary peer-checked:border-primary peer-checked:border-4 peer-focus-visible:ring-2 peer-focus-visible:ring-primary peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background transition-all" />
                          </div>
                          <span className="text-sm text-foreground">Male</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer group">
                          <div className="relative flex items-center justify-center w-5 h-5">
                            <input
                              type="radio"
                              name="gender"
                              value="1"
                              checked={gender === 1}
                              onChange={() => setGender(1)}
                              className="peer sr-only"
                            />
                            <div className="w-4 h-4 rounded-full border border-border group-hover:border-primary peer-checked:border-primary peer-checked:border-4 peer-focus-visible:ring-2 peer-focus-visible:ring-primary peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background transition-all" />
                          </div>
                          <span className="text-sm text-foreground">Female</span>
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-4 border-t border-border/50">
                    {QUESTIONNAIRE_SECTIONS.map((s, i) => (
                      <div key={i} className="bg-secondary/30 border border-border/40 rounded-lg p-3 text-center">
                        <div className="text-xs font-medium text-muted-foreground mb-1">{s.questions.length} questions</div>
                        <div className="text-sm font-semibold text-foreground leading-tight">{s.title.split("(")[0].trim()}</div>
                      </div>
                    ))}
                  </div>

                  <Button
                    onClick={() => setStep(1)}
                    disabled={!canProceed}
                    className="w-full rounded-full h-12 text-base shadow-md hover:shadow-lg transition-all"
                  >
                    Begin Assessment <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Breathing Transition */}
          {!isSubmitting && showTransition && (
            <motion.div
              key="breathing-transition"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="card-surface p-8 text-center space-y-6 max-w-md mx-auto"
            >
              <div className="text-primary font-semibold tracking-wide text-xs uppercase">
                Section {step} of {QUESTIONNAIRE_SECTIONS.length} Completed
              </div>
              <h3 className="text-2xl font-display font-bold text-foreground">Take a Mindful Breath</h3>
              
              <div className="flex items-center justify-center py-6">
                <motion.div
                  animate={{
                    scale: [1, 1.3, 1],
                    opacity: [0.6, 1, 0.6],
                  }}
                  transition={{
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                  className="w-28 h-28 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shadow-lg shadow-primary/5"
                >
                  <motion.div
                    animate={{
                      scale: [1, 1.15, 1],
                    }}
                    transition={{
                      duration: 4,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                    className="w-20 h-20 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-xs font-semibold text-primary"
                  >
                    Breathe
                  </motion.div>
                </motion.div>
              </div>

              <p className="text-sm text-muted-foreground leading-relaxed">
                Let's pause for a moment. Next: <span className="font-semibold text-foreground">{QUESTIONNAIRE_SECTIONS[step]?.title.split("(")[0].trim()}</span>.
              </p>

              <Button
                onClick={proceedToNext}
                className="w-full rounded-full h-11 text-sm shadow-md transition-all mt-4"
              >
                I'm Ready to Continue <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </motion.div>
          )}

          {/* Questionnaire sections */}
          {!isSubmitting && !showTransition && currentSection && (
            <motion.div
              key={`section-${step}`}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <div className="mb-6">
                <h2 className="text-xl font-display font-bold">{currentSection.title}</h2>
                <p className="text-sm text-muted-foreground mt-1">{currentSection.description}</p>
              </div>

              <div className="space-y-3">
                {currentSection.questions.map((q, i) => renderQuestion(q, i))}
              </div>

              <div className="flex items-center justify-between mt-8">
                <Button
                  variant="outline"
                  onClick={() => setStep(step - 1)}
                  className="rounded-full"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" /> Back
                </Button>

                {step < QUESTIONNAIRE_SECTIONS.length ? (
                  <Button
                    onClick={handleNextSection}
                    disabled={!canProceed}
                    className="rounded-full px-8"
                  >
                    Next Section <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                ) : (
                  <Button
                    onClick={handleSubmit}
                    disabled={!canProceed || isSubmitting}
                    className="rounded-full px-8 bg-accent hover:bg-accent/90"
                  >
                    {isSubmitting ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <>
                        <CheckCircle2 className="w-4 h-4 mr-2" /> Submit Assessment
                      </>
                    )}
                  </Button>
                )}
              </div>
            </motion.div>
          )}

        </AnimatePresence>
        </div>
      </main>
      <FooterSection compact />
    </div>
  );
}

