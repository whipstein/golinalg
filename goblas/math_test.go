package goblas

import (
	"math"
	"testing"
)

func TestAbsint(t *testing.T) {
	for i := 0; i < 100; i++ {
		mult := 1
		if i%2 == 0 {
			mult = -1
		}
		valint := i * mult
		if got, want := absint(valint), int(math.Abs(float64(valint))); got != want {
			t.Errorf("absint: values do not match: expected %d got %d", want, got)
		}
	}
}

func TestMaxMinint(t *testing.T) {
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			var wantint int
			multi := 1
			if i%2 == 0 {
				multi = -1
			}
			valinti := i * multi
			multj := 1
			if j%2 == 0 {
				multj = -1
			}
			valintj := j * multj

			if valinti < valintj {
				wantint = valintj
			} else {
				wantint = valinti
			}
			if got := maxint(valinti, valintj); got != wantint {
				t.Errorf("maxint: values do not match: expected %d got %d", wantint, got)
			}

			if valinti > valintj {
				wantint = valintj
			} else {
				wantint = valinti
			}
			if got := minint(valinti, valintj); got != wantint {
				t.Errorf("maxint: values do not match: expected %d got %d", wantint, got)
			}
		}
	}
}
