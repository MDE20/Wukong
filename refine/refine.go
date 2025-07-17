package refine

import (
	"../wukong"
	"../predicate"

	"log"
)

type RefinerFunc func(freq int, trainTracks [][]wukong.Detection, predFunc predicate.Predicate, modelCfg map[string]string, cfg map[string]string) Refiner

type Refiner interface {
	Plan(valTracks [][]wukong.Detection, bounds float64) map[string]string

	// returns list of frame indexes that need to be checked
	// and list of tracks that may need to be further refined
	Step(tracks [][]wukong.Detection, seen []int) (needed []int, refined []int)
	Close()

	// 动态调整参数
	AdjustParameters(factor float64)
}

var PSRefiners = make(map[string]RefinerFunc)
var InterpRefiners = make(map[string]RefinerFunc)

func logTrackState(tracks map[int][]wukong.Detection, msg string) {
	for id, t := range tracks {
		log.Printf("[debug] %s - TrackID %d, Length %d", msg, id, len(t))
	}
}

// Incorporate detections (that are labeled with track IDs) into tracks.
func incorporate(tracks map[int][]wukong.Detection, detections []wukong.Detection) {
	for _, detection := range detections {
		trackID := detection.TrackID
		track := tracks[trackID]
		insertIdx := findInsertIndex(track, detection.FrameIdx)

		var newTrack []wukong.Detection
		newTrack = append(newTrack, track[0:insertIdx]...)
		newTrack = append(newTrack, detection)
		newTrack = append(newTrack, track[insertIdx:]...)
		tracks[trackID] = newTrack
	}
}

func findInsertIndex(track []wukong.Detection, frameIdx int) int {
	for i, d := range track {
		if d.FrameIdx < frameIdx {
			continue
		}
		return i
	}
	return len(track) // 插入末尾
}

// Runs refiners, given underlying detections that are labeled with track IDs.
// Returns list of frames examined, and the refined tracks.
func RunFake(refiners []Refiner, tracks [][]wukong.Detection, detections [][]wukong.Detection) ([]int, [][]wukong.Detection) {
	seen := make(map[int]bool)
	for _, track := range tracks {
		for _, detection := range track {
			seen[detection.FrameIdx] = true
		}
	}

	getSeenList := func() []int {
		var seenList []int
		for frameIdx := range seen {
			seenList = append(seenList, frameIdx)
		}
		return seenList
	}

	trackByID := make(map[int][]wukong.Detection)
	for _, track := range tracks {
		if len(track) == 0 {
			continue
		}
		trackByID[track[0].TrackID] = track
	}

	logTrackState(trackByID, "Initial Track Map")

	for _, r := range refiners {
		var pending []int
		for trackID := range trackByID {
			pending = append(pending, trackID)
		}

		iteration := 0
		for len(pending) > 0 {
			iteration++
			log.Printf("[RunFake] Iteration %d, pending %d tracks", iteration, len(pending))

			inTracks := make([][]wukong.Detection, len(pending))
			for i, trackID := range pending {
				inTracks[i] = trackByID[trackID]
			}
			needed, refined := r.Step(inTracks, getSeenList())
			log.Printf("[refine-runfake] Step result: %d needed frames, %d refined tracks", len(needed), len(refined))

			for _, frameIdx := range needed {
				if seen[frameIdx] {
					log.Printf("[RunFake] Frame %d already seen, skipping", frameIdx)
					continue
				}
				if frameIdx < 0 || frameIdx >= len(detections) {
					log.Printf("[RunFake] Frame %d out of range, skipping", frameIdx)
					seen[frameIdx] = true
					continue
				}
				log.Printf("[RunFake] Incorporating detections from Frame %d", frameIdx)
				incorporate(trackByID, detections[frameIdx])
				seen[frameIdx] = true
			}

			// map from refined to track IDs
			var nextPending []int
			for _, i := range refined {
				if i >= 0 && i < len(pending) {
					trackID := pending[i]
					nextPending = append(nextPending, trackID)
				} else {
					log.Printf("[RunFake] Refined index %d out of bounds", i)
				}
			}
			pending = nextPending
		}
	}

	var outTracks [][]wukong.Detection
	for _, track := range trackByID {
		outTracks = append(outTracks, track)
	}

	log.Printf("[RunFake] Done. Total seen frames: %d", len(seen))
	return getSeenList(), outTracks
}
