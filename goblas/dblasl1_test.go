package goblas

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestDblasLevel1(t *testing.T) {
	var i, k, kn, ksize, len, n, np1 int
	var err error
	sfac := 9.765625e-4
	_case := &common.combla._case
	fmt.Printf("\n***** DBLAS Level 1 Tests *****\n")

	for _, *_case = range []string{"Ddot", "Daxpy", "Drotg", "Drot", "Dcopy", "Dswap", "Dnrm2", "Dasum", "Dscal", "Idamax", "Drotmg", "Drotm"} {
		err = nil
		if *_case == "Drotg" || *_case == "Drotmg" {
			var d12, sa, sb, sc, ss float64
			da1 := []float64{0.3, 0.4, -0.3, -0.4, -0.3, 0.0, 0.0, 1.0}
			datrue := []float64{0.5, 0.5, 0.5, -0.5, -0.5, 0.0, 1.0, 1.0}
			db1 := []float64{0.4, 0.3, 0.4, 0.3, -0.4, 0.0, 1.0, 0.0}
			dbtrue := []float64{0.0, 0.6, 0.0, -0.6, 0.0, 0.0, 1.0, 0.0}
			dc1 := []float64{0.6, 0.8, -0.6, 0.8, 0.6, 1.0, 0.0, 1.0}
			ds1 := []float64{0.8, 0.6, 0.8, -0.6, 0.8, 0.0, 1.0, 0.0}
			dab := [][]float64{
				{0.1, 0.3, 1.2, 0.2},
				{0.7, 0.2, 0.6, 4.2},
				{0, 0, 0, 0},
				{4, -1, 2, 4},
				{6e-10, 2e-2, 1e5, 10},
				{4e10, 2e-2, 1e-5, 10},
				{2e-10, 4e-2, 1e5, 10},
				{2e10, 4e-2, 1e-5, 10},
				{4, -2, 8, 4},
			}
			//    true results for modified Givens
			dtrue := [][]float64{
				{0, 0, 1.3, 0.2, 0, 0, 0, 0.5, 0},
				{0, 0, 4.5, 4.2, 1, 0.5, 0, 0, 0},
				{0, 0, 0, 0, -2, 0, 0, 0, 0},
				{0, 0, 0, 4, -1, 0, 0, 0, 0},
				{0, 15e-3, 0, 10, -1, 0, -1e-4, 0, 1},
				{0, 0, 6144e-5, 10, -1, 4096, -1e6, 0, 1},
				{0, 0, 15, 10, -1, 5e-5, 0, 1, 0},
				{0, 0, 15, 10, -1, 5e5, -4096, 1, 4096e-6},
				{0, 0, 7, 4, 0, 0, -0.5, -0.25, 0},
			}
			dmb := mat.NewDrotMatrixBuilder()
			drot := dmb.Build()

			//                   4096 = 2 ** 12
			d12 = 4096
			dtrue[0][0] = 12.0 / 130.0
			dtrue[0][1] = 36.0 / 130.0
			dtrue[0][6] = -1.0 / 6.0
			dtrue[1][0] = 14.0 / 75.0
			dtrue[1][1] = 49.0 / 75.0
			dtrue[1][8] = 1.0 / 7.0
			dtrue[4][0] = 45e-11 * (d12 * d12)
			dtrue[4][2] = 4e5 / (3.0 * d12)
			dtrue[4][5] = 1.0 / d12
			dtrue[4][7] = 1e4 / (3.0 * d12)
			dtrue[5][0] = 4e10 / (1.5 * d12 * d12)
			dtrue[5][1] = 2e-2 / 1.5
			dtrue[5][7] = 5e-7 * d12
			dtrue[6][0] = 4.0 / 150.0
			dtrue[6][1] = (2e-10 / 1.5) * (d12 * d12)
			dtrue[6][6] = -dtrue[4][5]
			dtrue[6][8] = 1e4 / d12
			dtrue[7][0] = dtrue[6][0]
			dtrue[7][1] = 2e10 / (1.5 * d12 * d12)
			dtrue[8][0] = 32.0 / 7.0
			dtrue[8][1] = -16.0 / 7.0
			//
			//     Compute true values which cannot be prestored
			//     in decimal notation
			//
			dbtrue[0] = 1.0 / 0.6
			dbtrue[2] = -1.0 / 0.6
			dbtrue[4] = 1.0 / 0.6

			common.combla.n = 0
			for k = 1; k <= 8; k++ {
				n = k
				common.combla.n++
				if *_case == "Drotg" {
					sa = da1[k-1]
					sb = db1[k-1]
					sa, sb, sc, ss = Drotg(sa, sb, sc, ss)
					if ok := dcompare1(sa, datrue[k-1], datrue[k-1], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(sb, dbtrue[k-1], dbtrue[k-1], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(sc, dc1[k-1], dc1[k-1], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(ss, ds1[k-1], ds1[k-1], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
				} else if *_case == "Drotmg" {
					d1, d2, x1, y1 := dab[k-1][0], dab[k-1][1], dab[k-1][2], dab[k-1][3]
					d1, d2, x1, drot = Drotmg(d1, d2, x1, y1)
					if ok := dcompare1(d1, dtrue[k-1][0], dtrue[k-1][0], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(d2, dtrue[k-1][1], dtrue[k-1][1], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(x1, dtrue[k-1][2], dtrue[k-1][2], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := icompare1(drot.Flag, int(dtrue[k-1][4])); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(drot.H11, dtrue[k-1][5], dtrue[k-1][5], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(drot.H21, dtrue[k-1][6], dtrue[k-1][6], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(drot.H12, dtrue[k-1][7], dtrue[k-1][7], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare1(drot.H22, dtrue[k-1][8], dtrue[k-1][8], sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
				}
			}

			if err == nil {
				passL1()
			} else {
				t.Fail()
				fmt.Print(err)
			}
		} else if *_case == "Dnrm2" || *_case == "Dasum" || *_case == "Dscal" || *_case == "Idamax" {
			dtrue1 := []float64{0.0, 0.3, 0.5, 0.7, 0.6}
			dtrue3 := []float64{0.0, 0.3, 0.7, 1.1, 1.0}
			sa := [][]float64{
				{0.3, -1.0, 0.0, 1.0, 0.3},
				{0.3, 0.3, 0.3, 0.3, 0.3},
			}
			dtrue := vf(8)
			itrue2 := []int{0, 1, 2, 2, 3}
			dtrue5 := [][][]float64{
				{
					{0.10, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
					{-0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
					{0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
					{0.20, -0.60, 0.30, 5.0, 5.0, 5.0, 5.0, 5.0},
					{0.03, -0.09, 0.15, -0.03, 6.0, 6.0, 6.0, 6.0},
				},
				{
					{0.10, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0},
					{0.09, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0},
					{0.09, 2.0, -0.12, 2.0, 2.0, 2.0, 2.0, 2.0},
					{0.06, 3.0, -0.18, 5.0, 0.09, 2.0, 2.0, 2.0},
					{0.03, 4.0, -0.09, 6.0, -0.15, 7.0, -0.03, 3.0},
				},
			}
			dv := [][][]float64{
				{
					{0.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
					{0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
					{0.3, -0.4, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
					{0.2, -0.6, 0.3, 5.0, 5.0, 5.0, 5.0, 5.0},
					{0.1, -0.3, 0.5, -0.1, 6.0, 6.0, 6.0, 6.0},
				},
				{
					{0.1, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0},
					{0.3, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0},
					{0.3, 2.0, -0.4, 2.0, 2.0, 2.0, 2.0, 2.0},
					{0.2, 3.0, -0.6, 5.0, 0.3, 2.0, 2.0, 2.0},
					{0.1, 4.0, -0.3, 6.0, -0.5, 7.0, -0.1, 3.0},
				},
			}

			common.combla.n = 0
			for incx := 1; incx <= 2; incx++ {
				common.combla.incx = incx
				for np1 = 1; np1 <= 5; np1++ {
					n = np1 - 1
					len = 2 * max(n, 1)
					dx := vf(8, incx)
					for i := 1; i <= len; i++ {
						dx.Set(i-1, dv[incx-1][np1-1][i-1])
					}

					common.combla.n++
					if *_case == "Dnrm2" {
						if ok := dcompare1(Dnrm2(n, dx), dtrue1[np1-1], dtrue1[np1-1], sfac); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Dasum" {
						if ok := dcompare1(Dasum(n, dx), dtrue3[np1-1], dtrue3[np1-1], sfac); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Dscal" {
						Dscal(n, sa[incx-1][np1-1], dx)
						for i := 1; i <= len; i++ {
							dtrue.Set(i-1, dtrue5[incx-1][np1-1][i-1])
						}
						if ok := dcompare(len, dx, dtrue, dtrue, sfac); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Idamax" {
						if ok := icompare1(Idamax(n, dx), itrue2[np1-1]); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					}
				}
			}

			if err == nil {
				passL1()
			} else {
				t.Fail()
				fmt.Print(err)
			}
		} else if *_case == "Ddot" || *_case == "Daxpy" || *_case == "Dcopy" || *_case == "Dswap" || *_case == "Drotm" || *_case == "Ddsdot" {
			da := 0.3
			incxs := []int{1, 2, -2, -1}
			incys := []int{1, -2, 1, -2}
			dtemp := vf(5)
			dx1 := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
			dy1 := []float64{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8}
			dsize := vf(7)
			dsize1 := []float64{0.0, 0.3, 1.6, 3.2}
			dsize2 := [][]float64{
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17},
			}
			dtx := vf(10)
			dty := vf(10)
			ns := []int{0, 1, 2, 4}
			dpar := [][]float64{
				{-2, 0, 0, 0, 0},
				{-1, 2, -3, -4, 5},
				{0, 0, 2, -3, 0},
				{1, 5, 2, 0, -4},
			}
			dt7 := [][]float64{
				{0.0, 0.30, 0.21, 0.62},
				{0.0, 0.30, -0.07, 0.85},
				{0.0, 0.30, -0.79, -0.74},
				{0.0, 0.30, 0.33, 1.27},
			}
			dt10x := [][][]float64{
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, -0.9, 0.3, 0.7, 0.0, 0.0, 0.0},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0},
					{0.8, 0.1, -0.6, 0.8, 0.3, -0.3, 0.5},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{-0.9, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0},
					{0.7, 0.1, 0.3, 0.8, -0.9, -0.3, 0.5},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.5, 0.3, -0.6, 0.8, 0.0, 0.0, 0.0},
				},
			}
			dt10y := [][][]float64{
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.1, -0.5, 0.8, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{-0.5, -0.9, 0.6, 0.0, 0.0, 0.0, 0.0},
					{-0.4, -0.9, 0.9, 0.7, -0.5, 0.2, 0.6},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{-0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0},
					{-0.4, 0.9, -0.5, 0.6, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.6, -0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
					{0.6, -0.9, 0.1, 0.7, -0.5, 0.2, 0.8},
				},
			}
			dt8 := [][][]float64{
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, -0.87, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, -0.87, 0.15, 0.94, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.35, -0.9, 0.48, 0.0, 0.0, 0.0, 0.0},
					{0.38, -0.9, 0.57, 0.7, -0.75, 0.2, 0.98},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.35, -0.72, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.38, -0.63, 0.15, 0.88, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.68, -0.9, 0.33, 0.0, 0.0, 0.0, 0.0},
					{0.68, -0.9, 0.33, 0.7, -0.75, 0.2, 1.04},
				},
			}
			lens := []int{1, 1, 2, 4, 1, 1, 3, 7}
			dt19x := [][][]float64{
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{-0.8, 0, 0, 0, 0, 0, 0},
					{-0.9, 0, 0, 0, 0, 0, 0},
					{3.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, 0, 0, 0, 0, 0},
					{-0.8, 3.8, 0, 0, 0, 0, 0},
					{-0.9, 2.8, 0, 0, 0, 0, 0},
					{3.5, -0.4, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0.8, 0, 0, 0},
					{-0.8, 3.8, -2.2, -1.2, 0, 0, 0},
					{-0.9, 2.8, -1.4, -1.3, 0, 0, 0},
					{3.5, -0.4, -2.2, 4.7, 0, 0, 0},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{-0.8, 0, 0, 0, 0, 0, 0},
					{-0.9, 0, 0, 0, 0, 0, 0},
					{3.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0, 0, 0, 0},
					{0, 0.1, -3.0, 0, 0, 0, 0},
					{-0.3, 0.1, -2.0, 0, 0, 0, 0},
					{3.3, 0.1, -2.0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4},
					{-2.0, 0.1, 1.4, 0.8, 0.6, -0.3, -2.8},
					{-1.8, 0.1, 1.3, 0.8, 0, -0.3, -1.9},
					{3.8, 0.1, -3.1, 0.8, 4.8, -0.3, -1.5},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{-0.8, 0, 0, 0, 0, 0, 0},
					{-0.9, 0, 0, 0, 0, 0, 0},
					{3.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0, 0, 0, 0},
					{4.8, 0.1, -3.0, 0, 0, 0, 0},
					{3.3, 0.1, -2.0, 0, 0, 0, 0},
					{2.1, 0.1, -2.0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4},
					{-1.6, 0.1, -2.2, 0.8, 5.4, -0.3, -2.8},
					{-1.5, 0.1, -1.4, 0.8, 3.6, -0.3, -1.9},
					{3.7, 0.1, -2.2, 0.8, 3.6, -0.3, -1.5},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
					{0.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0, 0, 0, 0, 0, 0},
					{-0.8, 0, 0, 0, 0, 0, 0},
					{-0.9, 0, 0, 0, 0, 0, 0},
					{3.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, 0, 0, 0, 0, 0},
					{-0.8, -1.0, 0, 0, 0, 0, 0},
					{-0.9, -0.8, 0, 0, 0, 0, 0},
					{3.5, 0.8, 0, 0, 0, 0, 0},
				},
				{
					{0.6, 0.1, -0.5, 0.8, 0, 0, 0},
					{-0.8, -1.0, 1.4, -1.6, 0, 0, 0},
					{-0.9, -0.8, 1.3, -1.6, 0, 0, 0},
					{3.5, 0.8, -3.1, 4.8, 0, 0, 0},
				},
			}
			dt19y := [][][]float64{
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.7, 0, 0, 0, 0, 0, 0},
					{1.7, 0, 0, 0, 0, 0, 0},
					{-2.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0, 0, 0, 0, 0},
					{0.7, -4.8, 0, 0, 0, 0, 0},
					{1.7, -0.7, 0, 0, 0, 0, 0},
					{-2.6, 3.5, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0.7, 0, 0, 0},
					{0.7, -4.8, 3.0, 1.1, 0, 0, 0},
					{1.7, -0.7, -0.7, 2.3, 0, 0, 0},
					{-2.6, 3.5, -0.7, -3.6, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.7, 0, 0, 0, 0, 0, 0},
					{1.7, 0, 0, 0, 0, 0, 0},
					{-2.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0, 0, 0, 0},
					{4.0, -0.9, -0.3, 0, 0, 0, 0},
					{-0.5, -0.9, 1.5, 0, 0, 0, 0},
					{-1.5, -0.9, -1.8, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8},
					{3.7, -0.9, -1.2, 0.7, -1.5, 0.2, 2.2},
					{-0.3, -0.9, 2.1, 0.7, -1.6, 0.2, 2.0},
					{-1.6, -0.9, -2.1, 0.7, 2.9, 0.2, -3.8},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.7, 0, 0, 0, 0, 0, 0},
					{1.7, 0, 0, 0, 0, 0, 0},
					{-2.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0, 0, 0, 0, 0},
					{4.0, -6.3, 0, 0, 0, 0, 0},
					{-0.5, 0.3, 0, 0, 0, 0, 0},
					{-1.5, 3.0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0.7, 0, 0, 0},
					{3.7, -7.2, 3.0, 1.7, 0, 0, 0},
					{-0.3, 0.9, -0.7, 1.9, 0, 0, 0},
					{-1.6, 2.7, -0.7, -3.4, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.5, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, 0, 0, 0, 0, 0, 0},
					{0.7, 0, 0, 0, 0, 0, 0},
					{1.7, 0, 0, 0, 0, 0, 0},
					{-2.6, 0, 0, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0, 0, 0, 0},
					{0.7, -0.9, 1.2, 0, 0, 0, 0},
					{1.7, -0.9, 0.5, 0, 0, 0, 0},
					{-2.6, -0.9, -1.3, 0, 0, 0, 0},
				},
				{
					{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8},
					{0.7, -0.9, 1.2, 0.7, -1.5, 0.2, 1.6},
					{1.7, -0.9, 0.5, 0.7, -1.6, 0.2, 2.4},
					{-2.6, -0.9, -1.3, 0.7, 2.9, 0.2, -4.0},
				},
			}

			common.combla.n = 0
			for ki := 1; ki <= 4; ki++ {
				incx := incxs[ki-1]
				incy := incys[ki-1]
				mx := abs(incx)
				my := abs(incy)
				common.combla.incx, common.combla.incy = incx, incy
				dx := vf(7, incx)
				dy := vf(7, incy)

				for kn = 1; kn <= 4; kn++ {
					n = ns[kn-1]
					ksize = min(2, kn)
					lenx := lens[kn-1+(mx-1)*4]
					leny := lens[kn-1+(my-1)*4]
					ksize := min(2, kn)

					for i := 1; i <= 7; i++ {
						dx.Set(i-1, dx1[i-1])
						dy.Set(i-1, dy1[i-1])
					}

					common.combla.n++
					if *_case == "Ddot" {
						if ok := dcompare1(Ddot(n, dx, dy), dt7[ki-1][kn-1], dsize1[kn-1], sfac); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Daxpy" {
						Daxpy(n, da, dx, dy)
						for j := 1; j <= leny; j++ {
							dty.Set(j-1, dt8[ki-1][kn-1][j-1])
						}
						dsizet := vf(dty.Size)
						for j := 1; j <= dty.Size; j++ {
							dsizet.Set(j-1, dsize2[ksize-1][j-1])
						}
						if ok := dcompare(leny, dy, dty, dsizet, sfac); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Dcopy" {
						for i := 1; i <= 7; i++ {
							dty.Set(i-1, dt10y[ki-1][kn-1][i-1])
						}
						Dcopy(n, dx, dy)
						if ok := dcompare(leny, dy, dty, vdf(dsize2[0]), 1.0); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Dswap" {
						Dswap(n, dx, dy)
						for i := 1; i <= 7; i++ {
							dtx.Set(i-1, dt10x[ki-1][kn-1][i-1])
							dty.Set(i-1, dt10y[ki-1][kn-1][i-1])
						}
						if ok := dcompare(lenx, dx, dtx, vdf(dsize2[0]), 1.0); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
						if ok := dcompare(leny, dy, dty, vdf(dsize2[0]), 1.0); ok != nil {
							err = fmt.Errorf("%v%v", err, ok)
						}
					} else if *_case == "Drotm" {
						kni := kn + 4*(ki-1)
						for kpar := 1; kpar <= 4; kpar++ {
							for i := 1; i <= 7; i++ {
								dx.Set(i-1, dx1[i-1])
								dy.Set(i-1, dy1[i-1])
								dtx.Set(i-1, dt19x[kni-1][kpar-1][i-1])
								dty.Set(i-1, dt19y[kni-1][kpar-1][i-1])
							}

							for i := 1; i <= 5; i++ {
								dtemp.Set(i-1, dpar[kpar-1][i-1])
							}

							for i := 1; i <= lenx; i++ {
								dsize.Set(i-1, dtx.Get(i-1))
							}

							// see remark above about dt11x[0,1,6] and dt11x[4,2,7]
							if (kpar == 2) && (kni == 7) {
								dsize.Set(0, 2.4)
							}
							if (kpar == 3) && (kni == 8) {
								dsize.Set(4, 1.8)
							}
							drb := mat.NewDrotMatrixBuilder()
							drot := drb.Flag(int(dtemp.Get(0))).H([4]float64{dtemp.Get(1), dtemp.Get(2), dtemp.Get(3), dtemp.Get(4)}).Build()
							dxx := vdf(dx.Data, incx)
							dyy := vdf(dy.Data, incy)

							Drotm(n, dxx, dyy, drot)
							if ok := dcompare(lenx, dxx, dtx, dsize, sfac); ok != nil {
								err = fmt.Errorf("%v%v", err, ok)
							}
							if ok := dcompare(leny, dyy, dty, dty, sfac); ok != nil {
								err = fmt.Errorf("%v%v", err, ok)
							}
						}
					}
				}
			}

			if err == nil {
				passL1()
			} else {
				t.Fail()
				fmt.Print(err)
			}
		} else if *_case == "Drot" {
			dc := 0.8
			ds := 0.6
			incxs := []int{1, 2, -2, -1}
			incys := []int{1, -2, 1, -2}
			dx1 := []float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4}
			dy1 := []float64{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8}
			mwpc := vf(11)
			mwps := vf(11)
			mwpstx := vf(5)
			mwpsty := vf(5)
			mwpx := vf(5)
			mwpy := vf(5)
			lens := []int{1, 1, 2, 4, 1, 1, 3, 7}
			ns := []int{0, 1, 2, 4}
			dtx := vf(7)
			dty := vf(7)
			dt9x := [][][]float64{
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, -0.46, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, -0.46, -0.22, 1.06, 0.0, 0.0, 0.0},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.66, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0},
					{0.96, 0.1, -0.76, 0.8, 0.90, -0.3, -0.02},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{-0.06, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0},
					{0.90, 0.1, -0.22, 0.8, 0.18, -0.3, -0.02},
				},
				{
					{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.78, 0.26, -0.76, 1.12, 0.0, 0.0, 0.0},
				},
			}
			dt9y := [][][]float64{
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, -0.78, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, -0.78, 0.54, 0.08, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.7, -0.9, -0.12, 0.0, 0.0, 0.0, 0.0},
					{0.64, -0.9, -0.30, 0.7, -0.18, 0.2, 0.28},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.7, -1.08, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.64, -1.26, 0.54, 0.20, 0.0, 0.0, 0.0},
				},
				{
					{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					{0.04, -0.9, 0.18, 0.0, 0.0, 0.0, 0.0},
					{0.04, -0.9, 0.18, 0.7, -0.18, 0.2, 0.16},
				},
			}
			mwpinx := make([]int, 11)
			mwpiny := make([]int, 11)
			mwpn := make([]int, 11)
			mwptx := vf(11 * 5)
			mwpty := vf(11 * 5)
			dsize2 := [][]float64{
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17},
			}

			common.combla.n = 0
			for ki := 1; ki <= 4; ki++ {
				incx := incxs[ki-1]
				incy := incys[ki-1]
				mx := abs(incx)
				my := abs(incy)
				dx := vf(7, incx)
				dy := vf(7, incy)
				for kn = 1; kn <= 4; kn++ {
					common.combla.n++
					n = ns[kn-1]
					ksize = min(2, kn)
					lenx := lens[kn-1+(mx-1)*4]
					leny := lens[kn-1+(my-1)*4]

					for i := 1; i <= 7; i++ {
						dx.Set(i-1, dx1[i-1])
						dy.Set(i-1, dy1[i-1])
						dtx.Set(i-1, dt9x[ki-1][kn-1][i-1])
						dty.Set(i-1, dt9y[ki-1][kn-1][i-1])
					}
					Drot(n, dx, dy, dc, ds)
					dsize2m := vdf(dsize2[1][ksize-1 : dx.Size+ksize-1])
					if ok := dcompare(lenx, dx, dtx, dsize2m, sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
					if ok := dcompare(leny, dy, dty, dsize2m, sfac); ok != nil {
						err = fmt.Errorf("%v%v", err, ok)
					}
				}
			}

			mwpc.Set(0, 1)
			for i = 2; i <= 11; i++ {
				mwpc.Set(i-1, 0)
			}
			mwps.Set(0, 0)
			for i = 2; i <= 6; i++ {
				mwps.Set(i-1, 1)
			}
			for i = 7; i <= 11; i++ {
				mwps.Set(i-1, -1)
			}
			mwpinx[0] = 1
			mwpinx[1] = 1
			mwpinx[2] = 1
			mwpinx[3] = -1
			mwpinx[4] = 1
			mwpinx[5] = -1
			mwpinx[6] = 1
			mwpinx[7] = 1
			mwpinx[8] = -1
			mwpinx[9] = 1
			mwpinx[10] = -1
			mwpiny[0] = 1
			mwpiny[1] = 1
			mwpiny[2] = -1
			mwpiny[3] = -1
			mwpiny[4] = 2
			mwpiny[5] = 1
			mwpiny[6] = 1
			mwpiny[7] = -1
			mwpiny[8] = -1
			mwpiny[9] = 2
			mwpiny[10] = 1
			for i = 1; i <= 11; i++ {
				mwpn[i-1] = 5
			}
			mwpn[4] = 3
			mwpn[9] = 3
			for i = 1; i <= 5; i++ {
				_i := float64(i)
				mwpx.Set(i-1, _i)
				mwpy.Set(i-1, _i)
				mwptx.Set(0+(i-1)*11, _i)
				mwpty.Set(0+(i-1)*11, _i)
				mwptx.Set(1+(i-1)*11, _i)
				mwpty.Set(1+(i-1)*11, -_i)
				mwptx.Set(2+(i-1)*11, 6-_i)
				mwpty.Set(2+(i-1)*11, _i-6)
				mwptx.Set(3+(i-1)*11, _i)
				mwpty.Set(3+(i-1)*11, -_i)
				mwptx.Set(5+(i-1)*11, 6-_i)
				mwpty.Set(5+(i-1)*11, _i-6)
				mwptx.Set(6+(i-1)*11, -_i)
				mwpty.Set(6+(i-1)*11, _i)
				mwptx.Set(7+(i-1)*11, _i-6)
				mwpty.Set(7+(i-1)*11, 6-_i)
				mwptx.Set(8+(i-1)*11, -_i)
				mwpty.Set(8+(i-1)*11, _i)
				mwptx.Set(10+(i-1)*11, _i-6)
				mwpty.Set(10+(i-1)*11, 6-_i)
			}
			mwptx.Set(4+(0)*11, 1)
			mwptx.Set(4+(1)*11, 3)
			mwptx.Set(4+(2)*11, 5)
			mwptx.Set(4+(3)*11, 4)
			mwptx.Set(4+(4)*11, 5)
			mwpty.Set(4+(0)*11, -1)
			mwpty.Set(4+(1)*11, 2)
			mwpty.Set(4+(2)*11, -2)
			mwpty.Set(4+(3)*11, 4)
			mwpty.Set(4+(4)*11, -3)
			mwptx.Set(9+(0)*11, -1)
			mwptx.Set(9+(1)*11, -3)
			mwptx.Set(9+(2)*11, -5)
			mwptx.Set(9+(3)*11, 4)
			mwptx.Set(9+(4)*11, 5)
			mwpty.Set(9+(0)*11, 1)
			mwpty.Set(9+(1)*11, 2)
			mwpty.Set(9+(2)*11, 2)
			mwpty.Set(9+(3)*11, 4)
			mwpty.Set(9+(4)*11, 3)
			for i = 1; i <= 11; i++ {
				common.combla.n++
				incx := mwpinx[i-1]
				incy := mwpiny[i-1]
				copyx := vf(5, incx)
				copyy := vf(5, incy)
				for k := 1; k <= 5; k++ {
					copyx.Set(k-1, mwpx.Get(k-1))
					copyy.Set(k-1, mwpy.Get(k-1))
					mwpstx.Set(k-1, mwptx.Get(i-1+(k-1)*11))
					mwpsty.Set(k-1, mwpty.Get(i-1+(k-1)*11))
				}

				Drot(mwpn[i-1], copyx, copyy, mwpc.Get(i-1), mwps.Get(i-1))
				if ok := dcompare(5, copyx, mwpstx, mwpstx, sfac); ok != nil {
					err = fmt.Errorf("%v%v", err, ok)
				}
				if ok := dcompare(5, copyy, mwpsty, mwpsty, sfac); ok != nil {
					err = fmt.Errorf("%v%v", err, ok)
				}
			}

			if err == nil {
				passL1()
			} else {
				t.Fail()
				fmt.Print(err)
			}
		}
	}
}

func dcompare1(scomp1, strue1, ssize1, sfac float64) (err error) {
	scomp := vdf([]float64{scomp1})
	strue := vdf([]float64{strue1})
	ssize := vdf([]float64{ssize1})

	return dcompare(1, scomp, strue, ssize, sfac)
}

func dcompare(len int, scomp, strue, ssize *mat.Vector, sfac float64) (err error) {
	var sd float64
	_case := &common.combla._case
	incx := &common.combla.incx
	incy := &common.combla.incy
	n := &common.combla.n

	for i := 0; i < len; i++ {
		sd = scomp.Get(i) - strue.Get(i)
		if math.Abs(sfac*sd) > math.Abs(ssize.Get(i))*eps {
			//
			//                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I).
			//
			if err != nil {
				err = fmt.Errorf("%v\n---FAIL---\n   Case    N  Incx  Incy\n %6s %4d  %4d  %4d\n comp[%3d]  = %36.24f\n true[%3d]  = %36.24f\n difference = %36.24f\n size[%3d]  = %36.24f\n", err, *_case, *n, *incx, *incy, i, scomp.Get(i), i, strue.Get(i), sd, i, ssize.Get(i))
			} else {
				err = fmt.Errorf("\n---FAIL---\n   Case    N  Incx  Incy\n %6s %4d  %4d  %4d\n comp[%3d]  = %36.24f\n true[%3d]  = %36.24f\n difference = %36.24f\n size[%3d]  = %36.24f\n", *_case, *n, *incx, *incy, i, scomp.Get(i), i, strue.Get(i), sd, i, ssize.Get(i))
			}
		}
	}

	return
}

func icompare1(icomp int, itrue int) (err error) {
	var id int
	var _case *string = &common.combla._case
	var n *int = &common.combla.n

	if icomp == itrue {
		return
	}
	//
	//                            HERE ICOMP IS NOT EQUAL TO ITRUE.
	//
	id = icomp - itrue
	return fmt.Errorf("                                       FAIL\n\n case  n                                comp                                true     difference\n %5s%3d%36d%36d%12d\n", *_case, *n, icomp, itrue, id)
}

func BenchmarkDasum1(b *testing.B)       { benchmarkDasum(1, b) }
func BenchmarkDasum10(b *testing.B)      { benchmarkDasum(10, b) }
func BenchmarkDasum100(b *testing.B)     { benchmarkDasum(100, b) }
func BenchmarkDasum1000(b *testing.B)    { benchmarkDasum(1000, b) }
func BenchmarkDasum10000(b *testing.B)   { benchmarkDasum(10000, b) }
func BenchmarkDasum100000(b *testing.B)  { benchmarkDasum(100000, b) }
func BenchmarkDasum1000000(b *testing.B) { benchmarkDasum(1000000, b) }
func benchmarkDasum(n int, b *testing.B) {
	da := 0.4
	dx := vf(n, 1)
	for i := 0; i < n; i++ {
		dx.Set(i, da)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Dasum(n, dx)
		_ = x
	}
}

func BenchmarkDrot1(b *testing.B)       { benchmarkDrot(1, b) }
func BenchmarkDrot10(b *testing.B)      { benchmarkDrot(10, b) }
func BenchmarkDrot100(b *testing.B)     { benchmarkDrot(100, b) }
func BenchmarkDrot1000(b *testing.B)    { benchmarkDrot(1000, b) }
func BenchmarkDrot10000(b *testing.B)   { benchmarkDrot(10000, b) }
func BenchmarkDrot100000(b *testing.B)  { benchmarkDrot(100000, b) }
func BenchmarkDrot1000000(b *testing.B) { benchmarkDrot(1000000, b) }
func benchmarkDrot(n int, b *testing.B) {
	da := 0.4
	dx := vf(n, 1)
	dy := vf(n, 1)
	for i := 0; i < n; i++ {
		dx.Set(i, da)
		dy.Set(i, da)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Drot(n, dx, dy, da, da)
	}
}

func BenchmarkDrotm1(b *testing.B)       { benchmarkDrotm(1, b) }
func BenchmarkDrotm10(b *testing.B)      { benchmarkDrotm(10, b) }
func BenchmarkDrotm100(b *testing.B)     { benchmarkDrotm(100, b) }
func BenchmarkDrotm1000(b *testing.B)    { benchmarkDrotm(1000, b) }
func BenchmarkDrotm10000(b *testing.B)   { benchmarkDrotm(10000, b) }
func BenchmarkDrotm100000(b *testing.B)  { benchmarkDrotm(100000, b) }
func BenchmarkDrotm1000000(b *testing.B) { benchmarkDrotm(1000000, b) }
func benchmarkDrotm(n int, b *testing.B) {
	da := 0.4
	dx := vf(n, 1)
	dy := vf(n, 1)
	for i := 0; i < n; i++ {
		dx.Set(i, da)
		dy.Set(i, da)
	}
	drb := mat.NewDrotMatrixBuilder()
	drot := drb.Flag(1).H([4]float64{5, 2, 0, -4}).Build()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Drotm(n, dx, dy, drot)
	}
}

func BenchmarkDscal1(b *testing.B)       { benchmarkDscal(1, b) }
func BenchmarkDscal10(b *testing.B)      { benchmarkDscal(10, b) }
func BenchmarkDscal100(b *testing.B)     { benchmarkDscal(100, b) }
func BenchmarkDscal1000(b *testing.B)    { benchmarkDscal(1000, b) }
func BenchmarkDscal10000(b *testing.B)   { benchmarkDscal(10000, b) }
func BenchmarkDscal100000(b *testing.B)  { benchmarkDscal(100000, b) }
func BenchmarkDscal1000000(b *testing.B) { benchmarkDscal(1000000, b) }
func benchmarkDscal(n int, b *testing.B) {
	sa := 0.3
	ca := 0.4
	cx := vf(n, 1)
	for i := 0; i < n; i++ {
		cx.Set(i, ca)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dscal(n, sa, cx)
	}
}

func BenchmarkDswap1(b *testing.B)       { benchmarkDswap(1, b) }
func BenchmarkDswap10(b *testing.B)      { benchmarkDswap(10, b) }
func BenchmarkDswap100(b *testing.B)     { benchmarkDswap(100, b) }
func BenchmarkDswap1000(b *testing.B)    { benchmarkDswap(1000, b) }
func BenchmarkDswap10000(b *testing.B)   { benchmarkDswap(10000, b) }
func BenchmarkDswap100000(b *testing.B)  { benchmarkDswap(100000, b) }
func BenchmarkDswap1000000(b *testing.B) { benchmarkDswap(1000000, b) }
func benchmarkDswap(n int, b *testing.B) {
	da := 0.4
	db := 0.8
	dx := vf(n, 1)
	dy := vf(n, 1)
	for i := 0; i < n; i++ {
		dx.Set(i, da)
		dy.Set(i, db)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dswap(n, dx, dy)
	}
}
