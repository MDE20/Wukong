package main

import (
	"../wukong"
	"../wukong/data"
	"../planner"

	"fmt"
	"log"
	"os"
	"strconv"
)

func planmain() {
	predName := os.Args[1]
	freq, _ := strconv.Atoi(os.Args[2])
	bound, _ := strconv.ParseFloat(os.Args[3], 64)

	var existingPlan wukong.PlannerConfig
	var qSamples map[int][]float64
	if len(os.Args) >= 5 {
		wukong.ReadJSON(os.Args[4], &existingPlan)
		qSamples = existingPlan.QSamples
	}

	ppCfg, modelCfg := data.Get(predName)
	log.Printf("[main] modelCfg: %+v\n", modelCfg)

	if qSamples == nil {
		qSamples = planner.GetQSamples(2*freq, ppCfg, modelCfg)
	}
	q := planner.PlanQ(qSamples, bound)
	log.Println("finished planning q", q)
	plan := wukong.PlannerConfig{
		Freq:     freq,
		Bound:    bound,
		QSamples: qSamples,
		Q:        q,
	}
	wukong.WriteJSON(fmt.Sprintf("logs/%s/%d/%v/plan.json", predName, freq, bound), plan)
	filterPlan, refinePlan := planner.PlanFilterRefine(ppCfg, modelCfg, freq, bound, nil)
	plan.Filter = filterPlan
	plan.Refine = refinePlan
	log.Printf("[main] filterPlan: %+v\n", filterPlan)
	log.Printf("[main] refinePlan: %+v\n", refinePlan)
	log.Println(plan)
	wukong.WriteJSON(fmt.Sprintf("logs/%s/%d/%v/plan.json", predName, freq, bound), plan)
}
