package main

import (
	"../wukong"
	"../predicate"

	"fmt"
	"os"
)

func getcountmain() {
	predName := os.Args[1]
	trackFname := os.Args[2]

	predFunc := predicate.GetPredicate(predName)
	var detections [][]wukong.Detection
	wukong.ReadJSON(trackFname, &detections)
	tracks := wukong.GetTracks(detections)
	var count int = 0
	for _, track := range tracks {
		if predFunc([][]wukong.Detection{track}) {
			count++
		}
	}
	fmt.Println(predName, count)
}
