package mat

// func TestMatrixDenseBuilder(t *testing.T) {
// 	mb := NewMatrixDenseBuilder()
// 	for _, stval := range []Style{General, Symmetric} {
// 		x := mb.Style(stval).Data(make([]float64, 30)).Size(2, 3).Major(Col).Build()
// 		for i := 0; i < 2; i++ {
// 			for j := 0; j < 3; j++ {
// 				x.Set(i, j, float64(i+j*2))
// 			}
// 		}

// 		xrows, xcols := x.Size()
// 		if len(x.Data()) != 6 || len(x.Data()) != xrows*xcols || xrows != 2 || xcols != 3 {
// 			t.Errorf("MatrixBuilder failed\n")
// 		}

// 		z := mb.Reset().Copy(x).Build()                                                    // Should not change underlying data of original matrix by setting new matrix
// 		zz := mb.Reset().Style(stval).Data(x.Data()).Size(2, 3).Major(Col).Build()         // Should not change underlying data of original matrix by setting new matrix
// 		zzz := mb.Reset().Style(stval).DataAddress(x.Data()).Size(2, 3).Major(Col).Build() // Should change underlying data of original matrix by setting new matrix
// 		zrows, _ := z.Size()
// 		for i := 1; i <= zrows; i++ {
// 			z.Set(i-1, 0, 0.5*float64(i))
// 			zz.Set(i-1, 0, 0.2*float64(i))
// 			zzz.Set(i-1, 0, 2.5*float64(i))
// 			fmt.Print()
// 		}
// 	}
// }

// func TestMatrixFactory(t *testing.T) {
// 	mf := MatrixFactory()
// 	x := mf(2, 3)
// 	for i := 0; i < 2; i++ {
// 		for j := 0; j < 3; j++ {
// 			x.Set(i, j, float64(i+j*2))
// 		}
// 	}

// 	xrows, xcols := x.Size()
// 	if len(x.Data()) != 6 || len(x.Data()) != xrows*xcols || xrows != 2 || xcols != 3 {
// 		t.Errorf("MatrixFactory failed\n")
// 	}

// 	mdf := MatrixDataFactory()
// 	mcpf := NewCopyMatrixFactory()
// 	z := mcpf(x)            // Should not change underlying data of original matrix by setting new matrix
// 	zz := mcpf(x)           // Should not change underlying data of original matrix by setting new matrix
// 	zzz := mdf(x, x.Data()) // Should change underlying data of original matrix by setting new matrix
// 	zrows, _ := z.Size()
// 	for i := 1; i <= zrows; i++ {
// 		z.Set(i-1, 0, 0.5*float64(i))
// 		zz.Set(i-1, 0, 0.2*float64(i))
// 		zzz.Set(i-1, 0, 2.5*float64(i))
// 		fmt.Print()
// 	}
// }

// func TestTransposeBuilder(t *testing.T) {
// 	mb := NewMatrixDenseBuilder()
// 	x := mb.Data(make([]float64, 30)).Size(5, 6).Major(Col).Build()
// 	for i := 0; i < 5; i++ {
// 		for j := 0; j < 6; j++ {
// 			x.Set(i, j, float64(i+j*5))
// 		}
// 	}

// 	xrows, xcols := x.Size()
// 	if len(x.Data()) != 30 || len(x.Data()) != xrows*xcols || xrows != 5 || xcols != 6 {
// 		t.Errorf("MatrixBuilder failed\n")
// 	}

// 	mat := x.T()
// 	if len(mat) != 30 {
// 		t.Errorf("TransposeBuilder failed\n")
// 	}
// 	for i := 0; i < 5; i++ {
// 		for j := 0; j < 6; j++ {
// 			if x.Get(i, j) != mat[i+j*5] {
// 				t.Errorf("TransposeBuilder failed: got %v  want %v\n", x.Get(i, j), mat[j+i*6])
// 			}
// 		}
// 	}
// }
