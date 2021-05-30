package mat

import (
	"testing"
)

// func TestVectorBuilder(t *testing.T) {
// 	vb := NewVectorBuilder()
// 	x := vb.Data(make([]float64, 10)).Build()
// 	for i := 0; i < x.Size(); i++ {
// 		x.Set(i, float64(i))
// 	}

// 	if x.Size() != 10 {
// 		t.Errorf("VectorBuilder failed\n")
// 	}

// 	y := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
// 	x = vb.Data(y).Build()

// 	for i, val := range y {
// 		if x.Get(i) != val {
// 			t.Errorf("VectorBuilder failed: got %v  want %v\n", x.Get(i), val)
// 		}
// 	}
// }

func TestVectorFactory(t *testing.T) {
	vf := VectorFactory()
	vdf := VectorDataFactory()
	x := vf(10)
	for i := 0; i < x.Size; i++ {
		x.Set(i, float64(i))
	}

	if x.Size != 10 {
		t.Errorf("VectorFactory failed\n")
	}

	y := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
	x = vdf(y)
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		x.Set(i, 0.5*x.Get(i))
	}
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
	}

	// x = vcf(y)
	for i, val := range y {
		if x.Get(i) != val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
		x.Set(i, 10*x.Get(i))
	}
	for i, val := range y {
		if x.Get(i) == val {
			t.Errorf("VectorFactory failed: got %v  want %v\n", x.Get(i), val)
		}
	}
}