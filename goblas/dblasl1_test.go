package goblas

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestDblasLevel1(t *testing.T) {
	var i, k, kn, ksize, len, n, np1 int
	sfac := 9.765625e-4
	_case := &common.combla._case
	pass := &common.combla.pass

	for _, *_case = range []string{"Ddot", "Daxpy", "Drotg", "Drot", "Dcopy", "Dswap", "Dnrm2", "Dasum", "Dscal", "Idamax", "Drotmg", "Drotm"} {
		*pass = true
		if *_case == "Drotg" || *_case == "Drotmg" {
			var d12, sa, sb, sc, ss float64
			da1 := vdf([]float64{0.3, 0.4, -0.3, -0.4, -0.3, 0.0, 0.0, 1.0})
			datrue := vdf([]float64{0.5, 0.5, 0.5, -0.5, -0.5, 0.0, 1.0, 1.0})
			db1 := vdf([]float64{0.4, 0.3, 0.4, 0.3, -0.4, 0.0, 1.0, 0.0})
			dbtrue := vdf([]float64{0.0, 0.6, 0.0, -0.6, 0.0, 0.0, 1.0, 0.0})
			dc1 := vdf([]float64{0.6, 0.8, -0.6, 0.8, 0.6, 1.0, 0.0, 1.0})
			ds1 := vdf([]float64{0.8, 0.6, 0.8, -0.6, 0.8, 0.0, 1.0, 0.0})
			dab := vdf([]float64{0.1, 0.3, 1.2, 0.2, 0.7, 0.2, 0.6, 4.2, 0, 0, 0, 0, 4, -1, 2, 4, 6e-10, 2e-2, 1e5, 10, 4e10, 2e-2, 1e-5, 10, 2e-10, 4e-2, 1e5, 10, 2e10, 4e-2, 1e-5, 10, 4, -2, 8, 4})
			//    true results for modified Givens
			dtrue := vdf([]float64{0, 0, 1.3, 0.2, 0, 0, 0, 0.5, 0, 0, 0, 4.5, 4.2, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 4, -1, 0, 0, 0, 0, 0, 15e-3, 0, 10, -1, 0, -1e-4, 0, 1, 0, 0, 6144e-5, 10, -1, 4096, -1e6, 0, 1, 0, 0, 15, 10, -1, 5e-5, 0, 1, 0, 0, 0, 15, 10, -1, 5e5, -4096, 1, 4096e-6, 0, 0, 7, 4, 0, 0, -0.5, -0.25, 0})
			dmb := mat.NewDrotMatrixBuilder()
			drot := dmb.Build()

			//                   4096 = 2 ** 12
			d12 = 4096
			dtrue.Set(0+0*9, 12.0/130.0)
			dtrue.Set(1+0*9, 36.0/130.0)
			dtrue.Set(6+0*9, -1.0/6.0)
			dtrue.Set(0+1*9, 14.0/75.0)
			dtrue.Set(1+1*9, 49.0/75.0)
			dtrue.Set(8+1*9, 1.0/7.0)
			dtrue.Set(0+4*9, 45e-11*(d12*d12))
			dtrue.Set(2+4*9, 4e5/(3.0*d12))
			dtrue.Set(5+4*9, 1.0/d12)
			dtrue.Set(7+4*9, 1e4/(3.0*d12))
			dtrue.Set(0+5*9, 4e10/(1.5*d12*d12))
			dtrue.Set(1+5*9, 2e-2/1.5)
			dtrue.Set(7+5*9, 5e-7*d12)
			dtrue.Set(0+6*9, 4.0/150.0)
			dtrue.Set(1+6*9, (2e-10/1.5)*(d12*d12))
			dtrue.Set(6+6*9, -dtrue.Get(5+4*9))
			dtrue.Set(8+6*9, 1e4/d12)
			dtrue.Set(0+7*9, dtrue.Get(0+6*9))
			dtrue.Set(1+7*9, 2e10/(1.5*d12*d12))
			dtrue.Set(0+8*9, 32.0/7.0)
			dtrue.Set(1+8*9, -16.0/7.0)
			//
			//     Compute true values which cannot be prestored
			//     in decimal notation
			//
			dbtrue.Set(0, 1.0/0.6)
			dbtrue.Set(2, -1.0/0.6)
			dbtrue.Set(4, 1.0/0.6)

			common.combla.n = 0
			for k = 1; k <= 8; k++ {
				n = k
				common.combla.n++
				if *_case == "Drotg" {
					sa = da1.Get(k - 1)
					sb = db1.Get(k - 1)
					Drotg(&sa, &sb, &sc, &ss)
					dcompare1(sa, datrue.Get(k-1), datrue.Get(k-1), sfac, t)
					dcompare1(sb, dbtrue.Get(k-1), dbtrue.Get(k-1), sfac, t)
					dcompare1(sc, dc1.Get(k-1), dc1.Get(k-1), sfac, t)
					dcompare1(ss, ds1.Get(k-1), ds1.Get(k-1), sfac, t)
				} else if *_case == "Drotmg" {
					d1, d2, x1, y1 := dab.Get(0+(k-1)*4), dab.Get(1+(k-1)*4), dab.Get(2+(k-1)*4), dab.Get(3+(k-1)*4)
					Drotmg(&d1, &d2, &x1, &y1, drot)
					dcompare1(d1, dtrue.Get(0+(k-1)*9), dtrue.Get(0+(k-1)*9), sfac, t)
					dcompare1(d2, dtrue.Get(1+(k-1)*9), dtrue.Get(1+(k-1)*9), sfac, t)
					dcompare1(x1, dtrue.Get(2+(k-1)*9), dtrue.Get(2+(k-1)*9), sfac, t)
					icompare1(drot.Flag, int(dtrue.Get(4+(k-1)*9)), t)
					dcompare1(drot.H11, dtrue.Get(5+(k-1)*9), dtrue.Get(5+(k-1)*9), sfac, t)
					dcompare1(drot.H21, dtrue.Get(6+(k-1)*9), dtrue.Get(6+(k-1)*9), sfac, t)
					dcompare1(drot.H12, dtrue.Get(7+(k-1)*9), dtrue.Get(7+(k-1)*9), sfac, t)
					dcompare1(drot.H22, dtrue.Get(8+(k-1)*9), dtrue.Get(8+(k-1)*9), sfac, t)
				}
			}

			if *pass {
				passL1()
			}
		} else if *_case == "Dnrm2" || *_case == "Dasum" || *_case == "Dscal" || *_case == "Idamax" {
			dtrue1 := vdf([]float64{0.0, 0.3, 0.5, 0.7, 0.6})
			dtrue3 := vdf([]float64{0.0, 0.3, 0.7, 1.1, 1.0})
			sa := vdf([]float64{0.3, -1.0, 0.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3})
			dtrue := vf(8)
			itrue2 := []int{0, 1, 2, 2, 3}
			dtrue5 := vdf([]float64{0.10, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.20, -0.60, 0.30, 5.0, 5.0, 5.0, 5.0, 5.0, 0.03, -0.09, 0.15, -0.03, 6.0, 6.0, 6.0, 6.0, 0.10, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.09, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.09, 2.0, -0.12, 2.0, 2.0, 2.0, 2.0, 2.0, 0.06, 3.0, -0.18, 5.0, 0.09, 2.0, 2.0, 2.0, 0.03, 4.0, -0.09, 6.0, -0.15, 7.0, -0.03, 3.0})
			dv := vdf([]float64{0.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.3, -0.4, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.2, -0.6, 0.3, 5.0, 5.0, 5.0, 5.0, 5.0, 0.1, -0.3, 0.5, -0.1, 6.0, 6.0, 6.0, 6.0, 0.1, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.3, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.3, 2.0, -0.4, 2.0, 2.0, 2.0, 2.0, 2.0, 0.2, 3.0, -0.6, 5.0, 0.3, 2.0, 2.0, 2.0, 0.1, 4.0, -0.3, 6.0, -0.5, 7.0, -0.1, 3.0})
			dx := vf(8)

			common.combla.n = 0
			for incx := 1; incx <= 2; incx++ {
				for np1 = 1; np1 <= 5; np1++ {
					n = np1 - 1
					len = 2 * maxint(n, 1)
					for i := 1; i <= len; i++ {
						dx.Set(i-1, dv.Get(i-1+(np1-1+(incx-1)*5)*8))
					}

					common.combla.n++
					if *_case == "Dnrm2" {
						dcompare1(Dnrm2(&n, dx, &incx), dtrue1.Get(np1-1), dtrue1.Get(np1-1), sfac, t)
					} else if *_case == "Dasum" {
						dcompare1(Dasum(&n, dx, &incx), dtrue3.Get(np1-1), dtrue3.Get(np1-1), sfac, t)
					} else if *_case == "Dscal" {
						Dscal(&n, sa.GetPtr(np1-1+(incx-1)*5), dx, &incx)
						for i := 1; i <= len; i++ {
							dtrue.Set(i-1, dtrue5.Get(i-1+(np1-1+(incx-1)*5)*8))
						}
						dcompare(len, dx, dtrue, dtrue, sfac, t)
					} else if *_case == "Idamax" {
						icompare1(Idamax(&n, dx, &incx), itrue2[np1-1], t)
					}
				}
			}

			if *pass {
				passL1()
			}
		} else if *_case == "Ddot" || *_case == "Daxpy" || *_case == "Dcopy" || *_case == "Dswap" || *_case == "Drotm" || *_case == "Ddsdot" {
			da := 0.3
			incxs := []int{1, 2, -2, -1}
			incys := []int{1, -2, 1, -2}
			dtemp := vf(5)
			dx1 := vdf([]float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4})
			dy1 := vdf([]float64{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8})
			dsize := vf(7)
			dsize1 := vdf([]float64{0.0, 0.3, 1.6, 3.2})
			dsize2 := vdf([]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17})
			dtx := vf(7)
			dty := vf(7)
			ns := []int{0, 1, 2, 4}
			dpar := vdf([]float64{-2, 0, 0, 0, 0, -1, 2, -3, -4, 5, 0, 0, 2, -3, 0, 1, 5, 2, 0, -4})
			dt7 := vdf([]float64{0.0, 0.30, 0.21, 0.62, 0.0, 0.30, -0.07, 0.85, 0.0, 0.30, -0.79, -0.74, 0.0, 0.30, 0.33, 1.27})
			dt10x := vdf([]float64{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.9, 0.3, 0.7, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, -0.6, 0.8, 0.3, -0.3, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.7, 0.1, 0.3, 0.8, -0.9, -0.3, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, -0.6, 0.8, 0.0, 0.0, 0.0})
			dt10y := vdf([]float64{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.1, -0.5, 0.8, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.9, 0.6, 0.0, 0.0, 0.0, 0.0, -0.4, -0.9, 0.9, 0.7, -0.5, 0.2, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.9, -0.5, 0.6, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, -0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.6, -0.9, 0.1, 0.7, -0.5, 0.2, 0.8})
			dt8 := vdf([]float64{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, -0.87, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, -0.87, 0.15, 0.94, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, -0.9, 0.48, 0.0, 0.0, 0.0, 0.0, 0.38, -0.9, 0.57, 0.7, -0.75, 0.2, 0.98, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, -0.72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38, -0.63, 0.15, 0.88, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68, -0.9, 0.33, 0.0, 0.0, 0.0, 0.0, 0.68, -0.9, 0.33, 0.7, -0.75, 0.2, 1.04})
			lens := []int{1, 1, 2, 4, 1, 1, 3, 7}
			dt19x := vdf([]float64{0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, -0.8, 0, 0, 0, 0, 0, 0, -0.9, 0, 0, 0, 0, 0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0.6, 0.1, 0, 0, 0, 0, 0, -0.8, 3.8, 0, 0, 0, 0, 0, -0.9, 2.8, 0, 0, 0, 0, 0, 3.5, -0.4, 0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0.8, 0, 0, 0, -0.8, 3.8, -2.2, -1.2, 0, 0, 0, -0.9, 2.8, -1.4, -1.3, 0, 0, 0, 3.5, -0.4, -2.2, 4.7, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, -0.8, 0, 0, 0, 0, 0, 0, -0.9, 0, 0, 0, 0, 0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0, 0, 0, 0, 0, 0.1, -3.0, 0, 0, 0, 0, -0.3, 0.1, -2.0, 0, 0, 0, 0, 3.3, 0.1, -2.0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4, -2.0, 0.1, 1.4, 0.8, 0.6, -0.3, -2.8, -1.8, 0.1, 1.3, 0.8, 0, -0.3, -1.9, 3.8, 0.1, -3.1, 0.8, 4.8, -0.3, -1.5, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, -0.8, 0, 0, 0, 0, 0, 0, -0.9, 0, 0, 0, 0, 0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0, 0, 0, 0, 4.8, 0.1, -3.0, 0, 0, 0, 0, 3.3, 0.1, -2.0, 0, 0, 0, 0, 2.1, 0.1, -2.0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4, -1.6, 0.1, -2.2, 0.8, 5.4, -0.3, -2.8, -1.5, 0.1, -1.4, 0.8, 3.6, -0.3, -1.9, 3.7, 0.1, -2.2, 0.8, 3.6, -0.3, -1.5, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, -0.8, 0, 0, 0, 0, 0, 0, -0.9, 0, 0, 0, 0, 0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0.6, 0.1, 0, 0, 0, 0, 0, -0.8, -1.0, 0, 0, 0, 0, 0, -0.9, -0.8, 0, 0, 0, 0, 0, 3.5, 0.8, 0, 0, 0, 0, 0, 0.6, 0.1, -0.5, 0.8, 0, 0, 0, -0.8, -1.0, 1.4, -1.6, 0, 0, 0, -0.9, -0.8, 1.3, -1.6, 0, 0, 0, 3.5, 0.8, -3.1, 4.8, 0, 0, 0})
			dt19y := vdf([]float64{0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 1.7, 0, 0, 0, 0, 0, 0, -2.6, 0, 0, 0, 0, 0, 0, 0.5, -0.9, 0, 0, 0, 0, 0, 0.7, -4.8, 0, 0, 0, 0, 0, 1.7, -0.7, 0, 0, 0, 0, 0, -2.6, 3.5, 0, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0.7, 0, 0, 0, 0.7, -4.8, 3.0, 1.1, 0, 0, 0, 1.7, -0.7, -0.7, 2.3, 0, 0, 0, -2.6, 3.5, -0.7, -3.6, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 1.7, 0, 0, 0, 0, 0, 0, -2.6, 0, 0, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0, 0, 0, 0, 4.0, -0.9, -0.3, 0, 0, 0, 0, -0.5, -0.9, 1.5, 0, 0, 0, 0, -1.5, -0.9, -1.8, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8, 3.7, -0.9, -1.2, 0.7, -1.5, 0.2, 2.2, -0.3, -0.9, 2.1, 0.7, -1.6, 0.2, 2.0, -1.6, -0.9, -2.1, 0.7, 2.9, 0.2, -3.8, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 1.7, 0, 0, 0, 0, 0, 0, -2.6, 0, 0, 0, 0, 0, 0, 0.5, -0.9, 0, 0, 0, 0, 0, 4.0, -6.3, 0, 0, 0, 0, 0, -0.5, 0.3, 0, 0, 0, 0, 0, -1.5, 3.0, 0, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0.7, 0, 0, 0, 3.7, -7.2, 3.0, 1.7, 0, 0, 0, -0.3, 0.9, -0.7, 1.9, 0, 0, 0, -1.6, 2.7, -0.7, -3.4, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 1.7, 0, 0, 0, 0, 0, 0, -2.6, 0, 0, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0, 0, 0, 0, 0.7, -0.9, 1.2, 0, 0, 0, 0, 1.7, -0.9, 0.5, 0, 0, 0, 0, -2.6, -0.9, -1.3, 0, 0, 0, 0, 0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8, 0.7, -0.9, 1.2, 0.7, -1.5, 0.2, 1.6, 1.7, -0.9, 0.5, 0.7, -1.6, 0.2, 2.4, -2.6, -0.9, -1.3, 0.7, 2.9, 0.2, -4.0})
			dx := vf(7)
			dy := vf(7)

			common.combla.n = 0
			for ki := 1; ki <= 4; ki++ {
				incx := incxs[ki-1]
				incy := incys[ki-1]
				mx := absint(incx)
				my := absint(incy)

				for kn = 1; kn <= 4; kn++ {
					n = ns[kn-1]
					ksize = minint(2, kn)
					lenx := lens[kn-1+(mx-1)*4]
					leny := lens[kn-1+(my-1)*4]
					ksize := minint(2, kn)

					for i := 1; i <= 7; i++ {
						dx.Set(i-1, dx1.Get(i-1))
						dy.Set(i-1, dy1.Get(i-1))
					}

					common.combla.n++
					if *_case == "Ddot" {
						dcompare1(Ddot(&n, dx, &incx, dy, &incy), dt7.Get(kn-1+(ki-1)*4), dsize1.Get(kn-1), sfac, t)
					} else if *_case == "Daxpy" {
						Daxpy(&n, &da, dx, &incx, dy, &incy)
						for j := 1; j <= leny; j++ {
							dty.Set(j-1, dt8.Get(j-1+(kn-1+(ki-1)*4)*7))
						}
						dsizet := vf(dty.Size)
						for j := 1; j <= dty.Size; j++ {
							dsizet.Set(j-1, dsize2.Get(j-1+(ksize-1)*14))
						}
						dcompare(leny, dy, dty, dsizet, sfac, t)
					} else if *_case == "Dcopy" {
						for i := 1; i <= 7; i++ {
							dty.Set(i-1, dt10y.Get(i-1+(kn-1+(ki-1)*4)*7))
						}
						Dcopy(&n, dx, &incx, dy, &incy)
						dcompare(leny, dy, dty, dsize2, 1.0, t)
					} else if *_case == "Dswap" {
						Dswap(&n, dx, &incx, dy, &incy)
						for i := 1; i <= 7; i++ {
							dtx.Set(i-1, dt10x.Get(i-1+(kn-1+(ki-1)*4)*7))
							dty.Set(i-1, dt10y.Get(i-1+(kn-1+(ki-1)*4)*7))
						}
						dcompare(lenx, dx, dtx, dsize2, 1.0, t)
						dcompare(leny, dy, dty, dsize2, 1.0, t)
					} else if *_case == "Drotm" {
						kni := kn + 4*(ki-1)
						for kpar := 1; kpar <= 4; kpar++ {
							for i := 1; i <= 7; i++ {
								dx.Set(i-1, dx1.Get(i-1))
								dy.Set(i-1, dy1.Get(i-1))
								dtx.Set(i-1, dt19x.Get(i-1+((kpar-1)+(kni-1)*4)*7))
								dty.Set(i-1, dt19y.Get(i-1+((kpar-1)+(kni-1)*4)*7))
							}

							for i := 1; i <= 5; i++ {
								dtemp.Set(i-1, dpar.Get(i-1+(kpar-1)*5))
							}

							for i := 1; i <= lenx; i++ {
								dsize.Set(i-1, dtx.Get(i-1))
							}
							//
							// see remark above about dt11x[0,1,6] and dt11x[4,2,7]
							//
							if (kpar == 2) && (kni == 7) {
								dsize.Set(0, 2.4)
							}
							if (kpar == 3) && (kni == 8) {
								dsize.Set(4, 1.8)
							}
							drb := mat.NewDrotMatrixBuilder()
							drot := drb.Flag(int(dtemp.Get(0))).H([4]float64{dtemp.Get(1), dtemp.Get(2), dtemp.Get(3), dtemp.Get(4)}).Build()

							Drotm(&n, dx, &incx, dy, &incy, drot)
							dcompare(lenx, dx, dtx, dsize, sfac, t)
							dcompare(leny, dy, dty, dty, sfac, t)
						}
					}
				}
			}

			if *pass {
				passL1()
			}
		} else if *_case == "Drot" {
			dc := 0.8
			ds := 0.6
			incxs := []int{1, 2, -2, -1}
			incys := []int{1, -2, 1, -2}
			copyx := vf(5)
			copyy := vf(5)
			dx := vf(7)
			dy := vf(7)
			dx1 := vdf([]float64{0.6, 0.1, -0.5, 0.8, 0.9, -0.3, -0.4})
			dy1 := vdf([]float64{0.5, -0.9, 0.3, 0.7, -0.6, 0.2, 0.8})
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
			dt9x := vdf([]float64{0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, -0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, -0.46, -0.22, 1.06, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.66, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.96, 0.1, -0.76, 0.8, 0.90, -0.3, -0.02, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.90, 0.1, -0.22, 0.8, 0.18, -0.3, -0.02, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.26, -0.76, 1.12, 0.0, 0.0, 0.0})
			dt9y := vdf([]float64{0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, -0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, -0.78, 0.54, 0.08, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, -0.9, -0.12, 0.0, 0.0, 0.0, 0.0, 0.64, -0.9, -0.30, 0.7, -0.18, 0.2, 0.28, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, -1.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.64, -1.26, 0.54, 0.20, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, -0.9, 0.18, 0.0, 0.0, 0.0, 0.0, 0.04, -0.9, 0.18, 0.7, -0.18, 0.2, 0.16})
			mwpinx := make([]int, 11)
			mwpiny := make([]int, 11)
			mwpn := make([]int, 11)
			mwptx := vf(11 * 5)
			mwpty := vf(11 * 5)
			dsize2 := vdf([]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17, 1.17})

			common.combla.n = 0
			for ki := 1; ki <= 4; ki++ {
				incx := incxs[ki-1]
				incy := incys[ki-1]
				mx := absint(incx)
				my := absint(incy)
				for kn = 1; kn <= 4; kn++ {
					common.combla.n++
					n = ns[kn-1]
					ksize = minint(2, kn)
					lenx := lens[kn-1+(mx-1)*4]
					leny := lens[kn-1+(my-1)*4]

					for i := 1; i <= 7; i++ {
						dx.Set(i-1, dx1.Get(i-1))
						dy.Set(i-1, dy1.Get(i-1))
						dtx.Set(i-1, dt9x.Get(i-1+(kn-1+(ki-1)*4)*7))
						dty.Set(i-1, dt9y.Get(i-1+(kn-1+(ki-1)*4)*7))
					}
					Drot(&n, dx, &incx, dy, &incy, &dc, &ds)
					dsize2m := vdf(dsize2.Data[ksize-1 : dx.Size+ksize-1])
					dcompare(lenx, dx, dtx, dsize2m, sfac, t)
					dcompare(leny, dy, dty, dsize2m, sfac, t)
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
				for k := 1; k <= 5; k++ {
					copyx.Set(k-1, mwpx.Get(k-1))
					copyy.Set(k-1, mwpy.Get(k-1))
					mwpstx.Set(k-1, mwptx.Get(i-1+(k-1)*11))
					mwpsty.Set(k-1, mwpty.Get(i-1+(k-1)*11))
				}
				Drot(&mwpn[i-1], copyx, &incx, copyy, &incy, mwpc.GetPtr(i-1), mwps.GetPtr(i-1))
				dcompare(5, copyx, mwpstx, mwpstx, sfac, t)
				dcompare(5, copyy, mwpsty, mwpsty, sfac, t)
			}

			if *pass {
				passL1()
			}
		}
	}
}

func dcompare1(scomp1, strue1, ssize1, sfac float64, t *testing.T) {
	scomp := vdf([]float64{scomp1})
	strue := vdf([]float64{strue1})
	ssize := vdf([]float64{ssize1})

	dcompare(1, scomp, strue, ssize, sfac, t)
}

func dcompare(len int, scomp, strue, ssize *mat.Vector, sfac float64, t *testing.T) {
	var sd float64
	pass := &common.combla.pass
	_case := &common.combla._case
	incx := &common.combla.incx
	incy := &common.combla.incy
	n := &common.combla.n

	for i := 0; i < len; i++ {
		sd = scomp.Get(i) - strue.Get(i)
		if math.Round(math.Abs(sfac*sd)) > math.Round(math.Abs(ssize.Get(i))*epsilonf64()) {
			//
			//                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I).
			//
			(*pass) = false
			t.Fail()
			fmt.Printf("                                       FAIL\n\n case  n incx incy  i                             comp[i]                             true[i]  difference     size[i]\n \n")
			fmt.Printf(" %6s%3d%5d%5d%3d%36.24f%36.24f%12.8f%12.8f\n", *_case, *n, *incx, *incy, i, scomp.Get(i), strue.Get(i), sd, ssize.Get(i))
		}
	}
}

func icompare1(icomp int, itrue int, t *testing.T) {
	var id int
	var _case *string = &common.combla._case
	var n *int = &common.combla.n
	var pass *bool = &common.combla.pass

	if icomp == itrue {
		return
	}
	//
	//                            HERE ICOMP IS NOT EQUAL TO ITRUE.
	//
	(*pass) = false
	id = icomp - itrue
	t.Fail()
	t.Logf("                                       FAIL\n\n case  n                                comp                                true     difference\n \n")
	t.Logf(" %5s%3d%36d%36d%12d\n", *_case, *n, icomp, itrue, id)
}
