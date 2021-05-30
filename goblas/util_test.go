package goblas

import "testing"

func TestMaxlocf32(t *testing.T) {
	var a []float32 = []float32{0.10, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.20, -0.60, 0.30, 5.0, 5.0, 5.0, 5.0, 5.0, 0.03, -0.09, 0.15, -0.03, 6.0, 6.0, 6.0, 6.0, 0.10, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.09, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.09, 2.0, -0.12, 2.0, 2.0, 2.0, 2.0, 2.0, 0.06, 3.0, -0.18, 5.0, 0.09, 2.0, 2.0, 2.0, 0.03, 4.0, -0.09, 6.0, -0.15, 7.0, -0.03, 3.0}
	for i := 0; i < 20; i++ {
		for j := 5; j < 80; j += 5 {
			if j > i {
				var cur float32
				var curidx int
				temp := a[i:j]
				for x, val := range temp {
					if x == 0 {
						cur = val
					} else if val > cur {
						cur = val
						curidx = x
					}
				}
				if got, want := maxlocf32(temp), curidx; got != want {
					t.Errorf("maxlocf32: values do not match: expected %d got %d", want, got)
				}
			}
		}
	}
}
