package main

import (
	"../wukong"
	"../data"
	rnnlib "../models/rnn"
	"../predicate"

	"fmt"
	"log"
	"os"
	"strconv"
)

// Prepares two datasets for RNN training:
// (1) Filter RNN: coarse tracks -> whether track satisfies the predicate
// (2) PS Refine RNN: coarse tracks -> whether track needs [prefix, suffix] refinement

func preparernnmain() {
	predName := os.Args[1]
	freq, _ := strconv.Atoi(os.Args[2])

	ppCfg, _ := data.Get(predName)

	predFunc := predicate.GetPredicate(ppCfg.Predicate)
	log.Printf("[prepare] loading train tracks")
	filterTrain, refineTrain := rnnlib.ItemsFromSegments(ppCfg.TrainSegments, freq, predFunc)
	log.Printf("[prepare] loading val tracks")
	filterVal, refineVal := rnnlib.ItemsFromSegments(ppCfg.ValSegments, freq, predFunc)

	wukong.WriteJSON(fmt.Sprintf("logs/%s/%d/filter_rnn_ds.json", predName, freq), rnnlib.DS{filterTrain, filterVal})
	wukong.WriteJSON(fmt.Sprintf("logs/%s/%d/refine_rnn_ds.json", predName, freq), rnnlib.DS{refineTrain, refineVal})
}
