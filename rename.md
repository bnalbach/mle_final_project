b1 > dqn-mode3-explore
b2 > dqn-mode3-tiebreak
b3 > dqn-mode3-hunt
b4 > dqn-mode3-4step
b4_hunt_safe > dqn-mode3-4step-s1
b4_danger_2rule > dqn-mode3-4step-s2
b4_full_3rule > dqn-mode3-4step-s3
b4_mixed_3opp > dqn-mode3-4step-s4


for all up to mode3 run: $python evaluate.py --agent {modelname} --rounds 100 
# custom agent it compares to doesnt matter

for all mode3 run: $python evaluate.py --agent {modelname} --rounds 100 --timeout 0.5 --out results/{modelname}.json 
# compare to the final_model for the report; add timeout for more realism; store results
# vs old model only makes sense if model is aleady very good (ignore the comparison for all but the last models: dqn-mode3-multi_opponents, dqn-mode3-???, dqn-mode3-???)
