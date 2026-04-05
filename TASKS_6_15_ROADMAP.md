# Tasks 6-15 Roadmap: Empirical Sweep & Scaling Evaluation

**Branch:** `feat/empirical-sweep-and-scaling`  
**Created:** April 4, 2026  
**Status:** Initialized (ready for Tasks 6-15)

---

## Phase Overview

This branch completes the research pipeline from theory (Tasks 1-5) to full empirical evaluation (Tasks 6-15).

### Phase 1: Empirical Sweep Pipeline (Tasks 6-10)
**Goal:** Generate complete abstention threshold empirical curves  
**Duration:** ~4-5 days  
**Deliverable:** Empirical vs theoretical comparison plots

### Phase 2: Scaling Evaluation (Tasks 11-13)  
**Goal:** Validate threshold functions on larger model  
**Duration:** 1-2 days  
**Deliverable:** 7B model sweep results

### Phase 3: Final Synthesis (Tasks 14-15)
**Goal:** Mathematical proof and summary report  
**Duration:** 2-3 days  
**Deliverable:** Formal proofs + midpoint report

---

## Task Breakdown

### Task 6: Build Sweep Pipeline Integration ⚙️
**Status:** Ready (sweep.py skeleton exists)  
**Work:**
- [ ] Integrate AbstractionSweep with pilot data
- [ ] Create sweep_pilot.py runner script
- [ ] Add result serialization to JSON
- [ ] Verify on 50-example pilot

**Files to Modify:**
- src/sweep.py (complete implementation)
- Create: sweep_pilot.py (integration script)

**Expected Output:**
- sweep_results_pilot_50.json (metrics per lambda)

---

### Task 7: Test Sweep on Pilot ✅
**Status:** Dependency on Task 6  
**Work:**
- [ ] Run full sweep on 50-example pilot
- [ ] Verify threshold calculations
- [ ] Check abstention vs accuracy tradeoffs
- [ ] Validate against theoretical predictions

**Expected Results:**
- Empirical lambdas optimal points identified
- Coverage vs accuracy tradeoff visualized

---

### Task 8: Scale to 500 Examples 📈
**Status:** Dependency on Task 7  
**Work:**
- [ ] Generate 500-example TriviaQA dataset
- [ ] Run batch inference (may take 6-8 hours)
- [ ] Save all predictions with confidence scores
- [ ] Compute accuracy ground truth

**Files to Create:**
- run_inference_full.py (adapted for 500 examples)

**Expected Output:**
- results/full_500_examples.json (500 inference results)

---

### Task 9: Run Full Sweep ⚡
**Status:** Dependency on Task 8  
**Work:**
- [ ] Execute AbstractionSweep on 500-example results
- [ ] Compute metrics across all lambdas (51 points)
- [ ] Store results with confidence intervals

**Expected Output:**
- sweep_results_full_500.json (complete metrics)

---

### Task 10: Generate Empirical Comparison 📊
**Status:** Dependency on Task 9  
**Work:**
- [ ] Create plot: Empirical vs theoretical thresholds
- [ ] Show accuracy/coverage tradeoff curves
- [ ] Add error bands (confidence intervals)
- [ ] Verify empirical matches theory

**Files to Modify:**
- gen_plots.py (add empirical plotting functions)

**Expected Output:**
- results/empirical_vs_theoretical.png
- results/tradeoff_curves.png

---

### Task 11: Test 7B Model Loading 🚀
**Status:** Optional/Experimental  
**Work:**
- [ ] Download Qwen2.5-7B-Instruct-GGUF in 4-bit quant
- [ ] Attempt to load with llama-server
- [ ] Record memory usage and loading time
- [ ] Test single inference call

**Risk:** May fail if GPU memory insufficient  
**Decision Point:** If fails → Skip Tasks 12-13

**Expected Output:**
- Server running with 7B model (if successful)

---

### Task 12: Generate 7B Inference (if Task 11 succeeds)
**Status:** Conditional on Task 11  
**Work:**
- [ ] Generate 200-example predictions with 7B model
- [ ] Compare confidence/accuracy vs 1.5B
- [ ] Save results for Task 13

**Expected Output:**
- results/qwen_7b_200_examples.json

---

### Task 13: Sweep on 7B Results (if Task 12 succeeds)
**Status:** Conditional on Task 12  
**Work:**
- [ ] Run sweep on 7B results (sample 3 key lambdas)
- [ ] Compare threshold effectiveness at scale
- [ ] Generate comparison plot

**Expected Output:**
- sweep_results_7b_sample.json
- results/comparison_1.5b_vs_7b.png

---

### Task 14: Formal Mathematical Proofs 📝
**Status:** Independent (can start anytime after Task 5)  
**Work:**
- [ ] Write proof: tau_U derivation
- [ ] Write proof: tau_B derivation  
- [ ] Write proof: tau_CE derivation
- [ ] Prove ordering: tau_CE ≥ tau_B ≥ tau_U

**Files to Create:**
- proofs/threshold_proofs.tex (LaTeX proofs)

**Expected Output:**
- proofs/threshold_proofs.pdf (compiled proofs)

---

### Task 15: Midpoint Report 📄
**Status:** Final task (requires Tasks 1-14)  
**Work:**
- [ ] Write 3-page main report:
  1. Problem statement & approach
  2. Theory summary (thresholds)
  3. Empirical results & comparison
- [ ] Add appendix:
  - Mathematical proofs
  - Data statistics
  - Implementation details
- [ ] Include all figures/plots

**Files to Create:**
- report/midpoint_report.md
- report/midpoint_report.pdf

**Expected Output:**
- Complete deliverable: midpoint_report.pdf (~3000 words)

---

## Dependency Graph

```
Tasks 1-5: Foundation (COMPLETE)
    ↓
Task 6: Sweep Integration (CRITICAL PATH)
    ↓
Task 7: Pilot Testing
    ↓
Tasks 8-10: Full Empirical Evaluation
    ↓
Task 11: 7B Scaling (Optional)
    ↓
Tasks 12-13: 7B Results (Conditional)
    ↓
Task 14: Proofs (Independent, parallel)
    ↓
Task 15: Final Report (Integration task)
```

---

## Key Dates & Milestones

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Task 6-7 Complete | April 5 | Pending |
| Tasks 8-10 Complete | April 7 | Pending |
| 7B Attempt (Task 11) | April 7 | Pending |
| Task 14 (Proofs) | April 8 | Pending |
| Task 15 (Report) | April 10 | Pending |
| **Midpoint Deadline** | **April 10** | **TARGET** |

---

## Critical Path Analysis

**Fastest Route to Completion:**
1. Tasks 6-7 (2 days) - Establish sweep works
2. Tasks 8-10 (3 days) - Generate empirical curves  
3. Task 15 (2 days) - Write report
4. **Total: 7 days** (April 4 → April 11)

**Buffer Available:** 1 day for adjustments/debugging

---

## Files to be Created/Modified

### New Files
- [ ] sweep_pilot.py - Pilot sweep runner
- [ ] run_inference_full.py - Full 500-ex inference
- [ ] proofs/threshold_proofs.tex - Mathematical proofs
- [ ] report/midpoint_report.md - Main report

### Modified Files
- [ ] src/sweep.py - Complete implementation
- [ ] gen_plots.py - Add empirical plotting
- [ ] src/config.py - Add new config parameters

### Results Generated
- [ ] results/sweep_results_pilot_50.json
- [ ] results/full_500_examples.json
- [ ] results/sweep_results_full_500.json
- [ ] results/empirical_vs_theoretical.png
- [ ] results/tradeoff_curves.png
- [ ] (Optional) results/qwen_7b_200_examples.json
- [ ] (Optional) results/comparison_1.5b_vs_7b.png

---

## Testing Strategy

### Unit Tests
- [ ] Test sweep.py on small dataset
- [ ] Verify threshold calculations
- [ ] Check result serialization

### Integration Tests
- [ ] Run full pipeline on 50-ex data
- [ ] Verify empirical matches theory (within error bounds)
- [ ] Check visualization outputs

### Validation Tests
- [ ] Compare 1.5B and 7B results (if both available)
- [ ] Verify mathematical proofs are sound
- [ ] Peer-review report content

---

## Success Criteria

### Tasks 6-10
- ✓ Sweep produces expected output
- ✓ Empirical results match theoretical predictions (within 5% error)
- ✓ Visualizations are clear and publication-ready

### Tasks 11-13
- ✓ 7B model loads successfully (or documented failure)
- ✓ Scaling analysis shows consistent threshold ordering
- ✓ Comparison plots generated

### Tasks 14-15
- ✓ Proofs are rigorous and complete
- ✓ Report is 3+ pages with clear narrative
- ✓ All figures referenced and explained
- ✓ Ready for submission by April 10

---

## Next Steps

**Immediate (Today - April 4):**
1. Review this roadmap
2. Begin Task 6: Complete sweep.py integration
3. Create sweep_pilot.py runner script

**Continue Reading:** Start with Task 6 implementation guide in next sections

---

**Branch Status:** ✓ Ready for work  
**Current Commit:** 1a52266  
**Last Updated:** April 4, 2026
