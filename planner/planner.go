package planner

import (
	"../wukong"
	"../predicate"

	"log"
)

type plannerContext struct {
	ppCfg           wukong.PreprocessConfig
	modelCfg        wukong.ModelConfig
	freq            int
	bound           float64
	trainDetections [][]wukong.Detection
	trainTracks     [][]wukong.Detection
	valDetections   [][]wukong.Detection
	valTracks       [][]wukong.Detection
	predFunc        predicate.Predicate
}

func PlanFilterRefine(ppCfg wukong.PreprocessConfig, modelCfg wukong.ModelConfig, freq int, bound float64, existingFilterPlan *wukong.FilterPlan) (wukong.FilterPlan, wukong.RefinePlan) {
	context := plannerContext{
		ppCfg:    ppCfg,
		modelCfg: modelCfg,
		freq:     freq,
		bound:    bound,
	}

	increment := func(detections [][]wukong.Detection, frames int, trackID int) {
		for _, dlist := range detections {
			for i := range dlist {
				dlist[i].FrameIdx += frames
				dlist[i].TrackID += trackID
			}
		}
	}
	getMaxTrackID := func(tracks [][]wukong.Detection) int {
		max := 0
		for _, track := range tracks {
			if track[0].TrackID > max {
				max = track[0].TrackID
			}
		}
		return max
	}

	context.predFunc = predicate.GetPredicate(ppCfg.Predicate)
	log.Printf("[planner] loading train tracks")
	for _, segment := range ppCfg.TrainSegments {
		segDetections := wukong.ReadDetections(segment.TrackPath)
		segTracks := wukong.GetTracks(segDetections)
		increment(segDetections, len(context.trainDetections), 0)
		increment(segTracks, len(context.trainDetections), getMaxTrackID(context.trainTracks)+1)
		context.trainDetections = append(context.trainDetections, segDetections...)
		context.trainTracks = append(context.trainTracks, segTracks...)
	}
	log.Printf("[planner] loading val tracks")
	for _, segment := range ppCfg.ValSegments {
		segDetections := wukong.ReadDetections(segment.TrackPath)
		segTracks := wukong.GetTracks(segDetections)
		increment(segDetections, len(context.valDetections), 0)
		increment(segTracks, len(context.valDetections), getMaxTrackID(context.valTracks)+1)
		context.valDetections = append(context.valDetections, segDetections...)
		context.valTracks = append(context.valTracks, segTracks...)
	}

	var filterPlan wukong.FilterPlan
	if existingFilterPlan != nil {
		filterPlan = *existingFilterPlan
	} else {
		filterPlan = PlanFilter(context)
	}
	//filterPlan := wukong.FilterPlan{"nnd", -18.007}
	refinePlan := PlanRefine(context, filterPlan)

	return filterPlan, refinePlan
}
