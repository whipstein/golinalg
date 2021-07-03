package goblas

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestZblasLevel1(t *testing.T) {
	var i, j, l, n, np1 int
	var err error
	sfac := 9.765625e-4
	ntest := &common.combla.n
	incx := &common.combla.incx
	incy := &common.combla.incy
	_case := &common.combla._case
	*incx = 0
	*incy = 0
	maxblocks := 4
	fmt.Printf("\n***** ZBLAS Level 1 Tests *****\n")

	for _, *_case = range []string{"Dznrm2", "Dzasum", "Zscal", "Zdscal", "Izamax", "Zdotc", "Zdotu", "Zaxpy", "Zcopy", "Zswap"} {
		err = nil
		n = 0
		*ntest = 0

		if *_case == "Dznrm2" || *_case == "Dzasum" || *_case == "Zscal" || *_case == "Zdscal" || *_case == "Izamax" {
			var ca complex128
			var sa float64
			cx := cvf(8)
			mwpcs := cvf(5)
			mwpct := cvf(5)

			sa, ca = 0.3, 0.4-0.7i
			cv := [][][]complex128{
				{
					{0.1 + 0.1i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i},
					{0.3 - 0.4i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i},
					{0.1 - 0.3i, 0.5 - 0.1i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i},
					{0.1 + 0.1i, -0.6 + 0.1i, 0.1 - 0.3i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i},
					{0.3 + 0.1i, 0.5 + 0.0i, 0.0 + 0.5i, 0.0 + 0.2i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i},
				},
				{
					{0.1 + 0.1i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i},
					{0.3 - 0.4i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i},
					{0.1 - 0.3i, 8.0 + 9.0i, 0.5 - 0.1i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i},
					{0.1 + 0.1i, 3.0 + 6.0i, -0.6 + 0.1i, 4.0 + 7.0i, 0.1 - 0.3i, 7.0 + 2.0i, 7.0 + 2.0i, 7.0 + 2.0i},
					{0.3 + 0.1i, 5.0 + 8.0i, 0.5 + 0.0i, 6.0 + 9.0i, 0.0 + 0.5i, 8.0 + 3.0i, 0.0 + 0.2i, 9.0 + 4.0i},
				},
			}
			strue2 := []float64{0.0, 0.5, 0.6, 0.7, 0.8}
			strue4 := []float64{0.0, 0.7, 1.0, 1.3, 1.6}
			ctrue5 := [][][]complex128{
				{
					{0.1 + 0.1i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i},
					{-0.16 - 0.37i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i},
					{-0.17 - 0.19i, 0.13 - 0.39i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i},
					{0.11 - 0.03i, -0.17 + 0.46i, -0.17 - 0.19i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i},
					{0.19 - 0.17i, 0.20 - 0.35i, 0.35 + 0.20i, 0.14 + 0.08i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i},
				},
				{
					{0.1 + 0.1i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i},
					{-0.16 - 0.37i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i},
					{-0.17 - 0.19i, 8.0 + 9.0i, 0.13 - 0.39i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i},
					{0.11 - 0.03i, 3.0 + 6.0i, -0.17 + 0.46i, 4.0 + 7.0i, -0.17 - 0.19i, 7.0 + 2.0i, 7.0 + 2.0i, 7.0 + 2.0i},
					{0.19 - 0.17i, 5.0 + 8.0i, 0.20 - 0.35i, 6.0 + 9.0i, 0.35 + 0.20i, 8.0 + 3.0i, 0.14 + 0.08i, 9.0 + 4.0i},
				},
			}
			ctrue6 := [][][]complex128{
				{
					{0.1 + 0.1i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i, 1.0 + 2.0i},
					{0.09 - 0.12i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i, 3.0 + 4.0i},
					{0.03 - 0.09i, 0.15 - 0.03i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i, 5.0 + 6.0i},
					{0.03 + 0.03i, -0.18 + 0.03i, 0.03 - 0.09i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i, 7.0 + 8.0i},
					{0.09 + 0.03i, 0.15 + 0.00i, 0.00 + 0.15i, 0.00 + 0.06i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i, 2.0 + 3.0i},
				},
				{
					{0.1 + 0.1i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i, 4.0 + 5.0i},
					{0.09 - 0.12i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i, 6.0 + 7.0i},
					{0.03 - 0.09i, 8.0 + 9.0i, 0.15 - 0.03i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i, 2.0 + 5.0i},
					{0.03 + 0.03i, 3.0 + 6.0i, -0.18 + 0.03i, 4.0 + 7.0i, 0.03 - 0.09i, 7.0 + 2.0i, 7.0 + 2.0i, 7.0 + 2.0i},
					{0.09 + 0.03i, 5.0 + 8.0i, 0.15 + 0.00i, 6.0 + 9.0i, 0.00 + 0.15i, 8.0 + 3.0i, 0.00 + 0.06i, 9.0 + 4.0i},
				},
			}
			itrue3 := []int{0, 1, 2, 2, 2}

			for (*incx) = 1; (*incx) <= 2; (*incx)++ {
				for np1 = 1; np1 <= 5; np1++ {
					n = np1 - 1
					l = 2 * max(n, 1)
					for i = 1; i <= l; i++ {
						cx.Set(i-1, cv[(*incx)-1][np1-1][i-1])
					}

					if *_case == "Dznrm2" {
						//              .. DZNRM2 ..
						if ok := dcompare1(Dznrm2(n, cx, *incx), strue2[np1-1], strue2[np1-1], sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Dzasum" {
						//              .. DZASUM ..
						if ok := dcompare1(Dzasum(n, cx, *incx), strue4[np1-1], strue4[np1-1], sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zscal" {
						//              .. ZSCAL ..
						Zscal(n, ca, cx, *incx)
						if ok := zcompare(l, cx, cvdf(ctrue5[(*incx)-1][np1-1]), cvdf(ctrue5[(*incx)-1][np1-1]), sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zdscal" {
						//              .. ZDSCAL ..
						Zdscal(n, sa, cx, *incx)
						if ok := zcompare(l, cx, cvdf(ctrue6[(*incx)-1][np1-1]), cvdf(ctrue6[(*incx)-1][np1-1]), sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Izamax" {
						//              .. IZAMAX ..
						if ok := icompare1(Izamax(n, cx, *incx), itrue3[np1-1]); ok != nil {
							err = addErr(err, ok)
						}
					}

					*ntest++
				}
			}

			if *_case == "Zscal" {
				for (*incx) = 1; (*incx) <= 2; (*incx)++ {
					for np1 = 2; np1 <= maxblocks+1; np1++ {
						n = np1 - 1
						zx := cvf(blocksize * minParBlocks * maxblocks)
						ctruex := cvf(blocksize * minParBlocks * maxblocks)
						for j = 1; j <= n; j++ {
							for i = 1; i <= n; i++ {
								if i < len(cv[(*incx)-1][np1-1]) {
									zx.Set((j-1)*blocksize+i-1, cv[(*incx)-1][np1-1][i-1])
									ctruex.Set((j-1)*blocksize+i-1, ctrue5[(*incx)-1][np1-1][i-1])
								}
							}
						}

						// //              .. ZSCAL ..
						Zscal(n*blocksize-n, ca, zx, *incx)
						if ok := zcompare(n*blocksize-n, zx, ctruex, ctruex, sfac); ok != nil {
							err = addErr(err, ok)
						}

						*ntest++
					}
				}
			}

			(*incx) = 1
			if *_case == "Zscal" {
				//        ZSCAL
				//        Add a test for alpha equal to zero.
				ca = 0.0 + 0.0i
				for i = 1; i <= 5; i++ {
					mwpct.Set(i-1, 0.0+0.0i)
					mwpcs.Set(i-1, 1.0+1.0i)
				}
				Zscal(5, ca, cx, *incx)
				if ok := zcompare(5, cx, mwpct, mwpcs, sfac); ok != nil {
					err = addErr(err, ok)
				}
				*ntest++
			} else if *_case == "Zdscal" {
				//        ZDSCAL
				//        Add a test for alpha equal to zero.
				sa = 0.0
				for i = 1; i <= 5; i++ {
					mwpct.Set(i-1, 0.0+0.0i)
					mwpcs.Set(i-1, 1.0+1.0i)
				}
				Zdscal(5, sa, cx, *incx)
				if ok := zcompare(5, cx, mwpct, mwpcs, sfac); ok != nil {
					err = addErr(err, ok)
				}
				*ntest++
				//        Add a test for alpha equal to one.
				sa = 1.0
				for i = 1; i <= 5; i++ {
					mwpct.Set(i-1, cx.Get(i-1))
					mwpcs.Set(i-1, cx.Get(i-1))
				}
				Zdscal(5, sa, cx, *incx)
				if ok := zcompare(5, cx, mwpct, mwpcs, sfac); ok != nil {
					err = addErr(err, ok)
				}
				*ntest++
				//        Add a test for alpha equal to minus one.
				sa = -1.0
				for i = 1; i <= 5; i++ {
					mwpct.Set(i-1, -cx.Get(i-1))
					mwpcs.Set(i-1, -cx.Get(i-1))
				}
				Zdscal(5, sa, cx, *incx)
				if ok := zcompare(5, cx, mwpct, mwpcs, sfac); ok != nil {
					err = addErr(err, ok)
				}
				*ntest++
			}

			if err == nil {
				passL1()
			} else {
				t.Fail()
				fmt.Print(err)
			}
		} else if *_case == "Zdotc" || *_case == "Zdotu" || *_case == "Zaxpy" || *_case == "Zcopy" || *_case == "Zswap" {
			var ki, kn, ksize, lenx, leny, mx, my int
			cdot := cvf(1)
			cx := cvf(8)
			cy := cvf(8)

			ca := 0.4 - 0.7i
			incxs := []int{1, 2, -2, -1}
			incys := []int{1, -2, 1, -2}
			lens := [][]int{
				{1, 1, 2, 4},
				{1, 1, 3, 7},
			}
			ns := []int{0, 1, 2, 4}
			cx1 := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
			cy1 := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
			ct8 := [][][]complex128{
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, -1.55 + 0.5i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, -1.55 + 0.5i, 0.03 - 0.89i, -0.38 - 0.96i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.07 - 0.89i, -0.9 + 0.5i, 0.42 - 1.41i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.78 + 0.06i, -0.9 + 0.5i, 0.06 - 0.13i, 0.1 - 0.5i, -0.77 - 0.49i, -0.5 - 0.3i, 0.52 - 1.51i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.07 - 0.89i, -1.18 - 0.31i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.78 + 0.06i, -1.54 + 0.97i, 0.03 - 0.89i, -0.18 - 1.31i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, -0.9 + 0.5i, 0.05 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.32 - 1.41i, -0.9 + 0.5i, 0.05 - 0.6i, 0.1 - 0.5i, -0.77 - 0.49i, -0.5 - 0.3i, 0.32 - 1.16i},
				},
			}
			ct7 := [][]complex128{
				{0.0 + 0.0i, -0.06 - 0.90i, 0.65 - 0.47i, -0.34 - 1.22i},
				{0.0 + 0.0i, -0.06 - 0.90i, -0.59 - 1.46i, -1.04 - 0.04i},
				{0.0 + 0.0i, -0.06 - 0.90i, -0.83 + 0.59i, 0.07 - 0.37i},
				{0.0 + 0.0i, -0.06 - 0.90i, -0.76 - 1.15i, -1.33 - 1.82i},
			}
			ct6 := [][]complex128{
				{0.0 + 0.0i, 0.90 + 0.06i, 0.91 - 0.77i, 1.80 - 0.10i},
				{0.0 + 0.0i, 0.90 + 0.06i, 1.45 + 0.74i, 0.20 + 0.90i},
				{0.0 + 0.0i, 0.90 + 0.06i, -0.55 + 0.23i, 0.83 - 0.39i},
				{0.0 + 0.0i, 0.90 + 0.06i, 1.04 + 0.79i, 1.95 + 1.22i},
			}
			ct10x := [][][]complex128{
				{
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, -0.9 + 0.5i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
				{
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.6i, -0.4 - 0.7i, 0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.8 - 0.7i, -0.4 - 0.7i, -0.1 - 0.2i, 0.2 - 0.8i, 0.7 - 0.6i, 0.1 + 0.4i, 0.6 - 0.6i},
				},
				{
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.9 + 0.5i, -0.4 - 0.7i, 0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.1 - 0.5i, -0.4 - 0.7i, 0.7 - 0.6i, 0.2 - 0.8i, -0.9 + 0.5i, 0.1 + 0.4i, 0.6 - 0.6i},
				},
				{
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.7 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.6 - 0.6i, 0.7 - 0.6i, -0.1 - 0.2i, 0.8 - 0.7i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
			}
			ct10y := [][][]complex128{
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, -0.4 - 0.7i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.1 - 0.9i, -0.9 + 0.5i, 0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.6 + 0.6i, -0.9 + 0.5i, -0.9 - 0.4i, 0.1 - 0.5i, -0.1 - 0.9i, -0.5 - 0.3i, 0.7 - 0.8i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.1 - 0.9i, 0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{-0.6 + 0.6i, -0.9 - 0.4i, -0.1 - 0.9i, 0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				},
				{
					{0.6 - 0.6i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, -0.9 + 0.5i, -0.4 - 0.7i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
					{0.7 - 0.8i, -0.9 + 0.5i, -0.4 - 0.7i, 0.1 - 0.5i, -0.1 - 0.9i, -0.5 - 0.3i, 0.2 - 0.8i},
				},
			}
			csize1 := []complex128{0.0 + 0.0i, 0.9 + 0.9i, 1.63 + 1.73i, 2.90 + 2.78i}
			csize3 := []complex128{0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 1.17 + 1.17i, 1.17 + 1.17i, 1.17 + 1.17i, 1.17 + 1.17i, 1.17 + 1.17i, 1.17 + 1.17i, 1.17 + 1.17i}
			csize2 := [][]complex128{
				{0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i},
				{1.54 + 1.54i, 1.54 + 1.54i, 1.54 + 1.54i, 1.54 + 1.54i, 1.54 + 1.54i, 1.54 + 1.54i, 1.54 + 1.54i},
			}

			for ki = 1; ki <= 4; ki++ {
				(*incx) = incxs[ki-1]
				(*incy) = incys[ki-1]
				mx = absint(*incx)
				my = absint(*incy)

				for kn = 1; kn <= 4; kn++ {
					n = ns[kn-1]
					ksize = min(2, kn)
					lenx = lens[mx-1][kn-1]
					leny = lens[my-1][kn-1]

					for i = 1; i <= 7; i++ {
						cx.Set(i-1, cx1[i-1])
						cy.Set(i-1, cy1[i-1])
					}

					if *_case == "Zdotc" {
						//              .. ZDOTC ..
						cdot.Set(0, Zdotc(n, cx, *incx, cy, *incy))
						if ok := zcompare(1, cdot, cvdf(ct6[ki-1][kn-1:]), cvdf(csize1[kn-1:]), sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zdotu" {
						//              .. ZDOTU ..
						cdot.Set(0, Zdotu(n, cx, *incx, cy, *incy))
						if ok := zcompare(1, cdot, cvdf(ct7[ki-1][kn-1:]), cvdf(csize1[kn-1:]), sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zaxpy" {
						//              .. ZAXPY ..
						Zaxpy(n, ca, cx, *incx, cy, *incy)
						if ok := zcompare(leny, cy, cvdf(ct8[ki-1][kn-1]), cvdf(csize2[ksize-1]), sfac); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zcopy" {
						//              .. ZCOPY ..
						Zcopy(n, cx, *incx, cy, *incy)
						if ok := zcompare(leny, cy, cvdf(ct10y[ki-1][kn-1]), cvdf(csize3), 1.0); ok != nil {
							err = addErr(err, ok)
						}
					} else if *_case == "Zswap" {
						//              .. ZSWAP ..
						Zswap(n, cx, *incx, cy, *incy)
						if ok := zcompare(lenx, cx, cvdf(ct10x[ki-1][kn-1]), cvdf(csize3), 1.0); ok != nil {
							err = addErr(err, ok)
						}
						if ok := zcompare(leny, cy, cvdf(ct10y[ki-1][kn-1]), cvdf(csize3), 1.0); ok != nil {
							err = addErr(err, ok)
						}
					}

					*ntest++
				}
			}

			if *_case == "Zaxpy" {
				for nx := 1; nx <= maxblocks; nx++ {
					for ki = 1; ki <= 1; ki++ {
						(*incx) = incxs[ki-1]
						(*incy) = incys[ki-1]
						mx = absint(*incx)
						my = absint(*incy)

						for kn = 1; kn <= 4; kn++ {
							n = ns[kn-1]
							ksize = min(2, kn)
							lenx = lens[mx-1][kn-1]
							leny = lens[my-1][kn-1]

							zx := cvf(blocksize * minParBlocks * maxblocks)
							zy := cvf(blocksize * minParBlocks * maxblocks)
							ctruex := cvf(blocksize * minParBlocks * maxblocks)
							csizex := cvf(blocksize * minParBlocks * maxblocks)
							for j = 1; j <= nx; j++ {
								for i = 1; i <= min(n*mx, len(cx1)); i++ {
									zx.Set((j-1)*blocksize+i-1, cx1[i-1])
								}
								for i = 1; i <= min(n*my, len(cy1)); i++ {
									zy.Set((j-1)*blocksize+i-1, cy1[i-1])
								}
								if n > 0 {
									for i = 1; i <= 7; i++ {
										ctruex.Set((j-1)*blocksize+i-1, ct8[ki-1][kn-1][i-1])
										csizex.Set((j-1)*blocksize+i-1, csize2[ksize-1][i-1])
									}
								}
							}

							Zaxpy((nx-1)*blocksize+n, ca, zx, *incx, zy, *incy)
							if ok := zcompare((nx-1)*blocksize+n, zy, ctruex, csizex, sfac); ok != nil {
								err = addErr(err, ok)
							}

							*ntest++
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
		}
	}
}

func zcompare(l int, ccomp, ctrue, csize *mat.CVector, sfac float64) (err error) {
	var i int

	scomp := vf(2 * l)
	ssize := vf(2 * l)
	strue := vf(2 * l)

	for i = 1; i <= l; i++ {
		scomp.Set(2*(i-1), real(ccomp.Get(i-1)))
		scomp.Set(2*(i-1)+1, imag(ccomp.Get(i-1)))
		strue.Set(2*(i-1), real(ctrue.Get(i-1)))
		strue.Set(2*(i-1)+1, imag(ctrue.Get(i-1)))
		ssize.Set(2*(i-1), real(csize.Get(i-1)))
		ssize.Set(2*(i-1)+1, imag(csize.Get(i-1)))
	}

	return dcompare(2*l, scomp, strue, ssize, sfac)
}

func addErr(err, ok error) error {
	if err != nil {
		return fmt.Errorf("%v%v", err, ok)
	}
	return fmt.Errorf("%v", ok)
}

func BenchmarkDzasum1(b *testing.B)       { benchmarkDzasum(1, b) }
func BenchmarkDzasum10(b *testing.B)      { benchmarkDzasum(10, b) }
func BenchmarkDzasum100(b *testing.B)     { benchmarkDzasum(100, b) }
func BenchmarkDzasum1000(b *testing.B)    { benchmarkDzasum(1000, b) }
func BenchmarkDzasum10000(b *testing.B)   { benchmarkDzasum(10000, b) }
func BenchmarkDzasum100000(b *testing.B)  { benchmarkDzasum(100000, b) }
func BenchmarkDzasum1000000(b *testing.B) { benchmarkDzasum(1000000, b) }
func benchmarkDzasum(n int, b *testing.B) {
	incx := 1
	ca := 0.4 - 0.7i
	cx := cvf(n)
	for i := 0; i < n; i++ {
		cx.Set(i, ca)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Dzasum(n, cx, incx)
		_ = x
	}
}

func BenchmarkDznrm2_1(b *testing.B)       { benchmarkDznrm2(1, b) }
func BenchmarkDznrm2_10(b *testing.B)      { benchmarkDznrm2(10, b) }
func BenchmarkDznrm2_100(b *testing.B)     { benchmarkDznrm2(100, b) }
func BenchmarkDznrm2_1000(b *testing.B)    { benchmarkDznrm2(1000, b) }
func BenchmarkDznrm2_10000(b *testing.B)   { benchmarkDznrm2(10000, b) }
func BenchmarkDznrm2_100000(b *testing.B)  { benchmarkDznrm2(100000, b) }
func BenchmarkDznrm2_1000000(b *testing.B) { benchmarkDznrm2(1000000, b) }
func benchmarkDznrm2(n int, b *testing.B) {
	incx := 1
	ca := 0.4 - 0.7i
	cx := cvf(n)
	for i := 0; i < n; i++ {
		cx.Set(i, ca)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Dznrm2(n, cx, incx)
		_ = x
	}
}

func BenchmarkZscal1(b *testing.B)       { benchmarkZscal(1, b) }
func BenchmarkZscal10(b *testing.B)      { benchmarkZscal(10, b) }
func BenchmarkZscal100(b *testing.B)     { benchmarkZscal(100, b) }
func BenchmarkZscal1000(b *testing.B)    { benchmarkZscal(1000, b) }
func BenchmarkZscal10000(b *testing.B)   { benchmarkZscal(10000, b) }
func BenchmarkZscal100000(b *testing.B)  { benchmarkZscal(100000, b) }
func BenchmarkZscal1000000(b *testing.B) { benchmarkZscal(1000000, b) }
func benchmarkZscal(n int, b *testing.B) {
	incx := 1
	ca := 0.4 - 0.7i
	cx := cvf(n)
	for i := 0; i < n; i++ {
		cx.Set(i, ca)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zscal(n, ca, cx, incx)
	}
}

func BenchmarkZdscal1(b *testing.B)       { benchmarkZdscal(1, b) }
func BenchmarkZdscal10(b *testing.B)      { benchmarkZdscal(10, b) }
func BenchmarkZdscal100(b *testing.B)     { benchmarkZdscal(100, b) }
func BenchmarkZdscal1000(b *testing.B)    { benchmarkZdscal(1000, b) }
func BenchmarkZdscal10000(b *testing.B)   { benchmarkZdscal(10000, b) }
func BenchmarkZdscal100000(b *testing.B)  { benchmarkZdscal(100000, b) }
func BenchmarkZdscal1000000(b *testing.B) { benchmarkZdscal(1000000, b) }
func benchmarkZdscal(n int, b *testing.B) {
	incx := 1
	sa := 0.3
	ca := 0.4 - 0.7i
	cx := cvf(n)
	for i := 0; i < n; i++ {
		cx.Set(i, ca)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zdscal(n, sa, cx, incx)
	}
}

func BenchmarkZdotc1(b *testing.B)       { benchmarkZdotc(1, b) }
func BenchmarkZdotc10(b *testing.B)      { benchmarkZdotc(10, b) }
func BenchmarkZdotc100(b *testing.B)     { benchmarkZdotc(100, b) }
func BenchmarkZdotc1000(b *testing.B)    { benchmarkZdotc(1000, b) }
func BenchmarkZdotc10000(b *testing.B)   { benchmarkZdotc(10000, b) }
func BenchmarkZdotc100000(b *testing.B)  { benchmarkZdotc(100000, b) }
func BenchmarkZdotc1000000(b *testing.B) { benchmarkZdotc(1000000, b) }
func benchmarkZdotc(n int, b *testing.B) {
	incx := 1
	incy := 1
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Zdotc(n, cx, incx, cy, incy)
		y := x
		_ = y
	}
}

func BenchmarkZdotu1(b *testing.B)       { benchmarkZdotu(1, b) }
func BenchmarkZdotu10(b *testing.B)      { benchmarkZdotu(10, b) }
func BenchmarkZdotu100(b *testing.B)     { benchmarkZdotu(100, b) }
func BenchmarkZdotu1000(b *testing.B)    { benchmarkZdotu(1000, b) }
func BenchmarkZdotu10000(b *testing.B)   { benchmarkZdotu(10000, b) }
func BenchmarkZdotu100000(b *testing.B)  { benchmarkZdotu(100000, b) }
func BenchmarkZdotu1000000(b *testing.B) { benchmarkZdotu(1000000, b) }
func benchmarkZdotu(n int, b *testing.B) {
	incx := 1
	incy := 1
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Zdotu(n, cx, incx, cy, incy)
		y := x
		_ = y
	}
}

func BenchmarkZaxpy1(b *testing.B)       { benchmarkZaxpy(1, b) }
func BenchmarkZaxpy10(b *testing.B)      { benchmarkZaxpy(10, b) }
func BenchmarkZaxpy100(b *testing.B)     { benchmarkZaxpy(100, b) }
func BenchmarkZaxpy1000(b *testing.B)    { benchmarkZaxpy(1000, b) }
func BenchmarkZaxpy10000(b *testing.B)   { benchmarkZaxpy(10000, b) }
func BenchmarkZaxpy100000(b *testing.B)  { benchmarkZaxpy(100000, b) }
func BenchmarkZaxpy1000000(b *testing.B) { benchmarkZaxpy(1000000, b) }
func benchmarkZaxpy(n int, b *testing.B) {
	incx := 1
	incy := 1
	ca := 0.4 - 0.7i
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zaxpy(n, ca, cx, incx, cy, incy)
	}
}

func BenchmarkZcopy1(b *testing.B)       { benchmarkZcopy(1, b) }
func BenchmarkZcopy10(b *testing.B)      { benchmarkZcopy(10, b) }
func BenchmarkZcopy100(b *testing.B)     { benchmarkZcopy(100, b) }
func BenchmarkZcopy1000(b *testing.B)    { benchmarkZcopy(1000, b) }
func BenchmarkZcopy10000(b *testing.B)   { benchmarkZcopy(10000, b) }
func BenchmarkZcopy100000(b *testing.B)  { benchmarkZcopy(100000, b) }
func BenchmarkZcopy1000000(b *testing.B) { benchmarkZcopy(1000000, b) }
func benchmarkZcopy(n int, b *testing.B) {
	incx := 1
	incy := 1
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zcopy(n, cx, incx, cy, incy)
	}
}

func BenchmarkZswap1(b *testing.B)       { benchmarkZswap(1, b) }
func BenchmarkZswap10(b *testing.B)      { benchmarkZswap(10, b) }
func BenchmarkZswap100(b *testing.B)     { benchmarkZswap(100, b) }
func BenchmarkZswap1000(b *testing.B)    { benchmarkZswap(1000, b) }
func BenchmarkZswap10000(b *testing.B)   { benchmarkZswap(10000, b) }
func BenchmarkZswap100000(b *testing.B)  { benchmarkZswap(100000, b) }
func BenchmarkZswap1000000(b *testing.B) { benchmarkZswap(1000000, b) }
func benchmarkZswap(n int, b *testing.B) {
	incx := 1
	incy := 1
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zswap(n, cx, incx, cy, incy)
	}
}

func BenchmarkZdrot1(b *testing.B)       { benchmarkZdrot(1, b) }
func BenchmarkZdrot10(b *testing.B)      { benchmarkZdrot(10, b) }
func BenchmarkZdrot100(b *testing.B)     { benchmarkZdrot(100, b) }
func BenchmarkZdrot1000(b *testing.B)    { benchmarkZdrot(1000, b) }
func BenchmarkZdrot10000(b *testing.B)   { benchmarkZdrot(10000, b) }
func BenchmarkZdrot100000(b *testing.B)  { benchmarkZdrot(100000, b) }
func BenchmarkZdrot1000000(b *testing.B) { benchmarkZdrot(1000000, b) }
func benchmarkZdrot(n int, b *testing.B) {
	incx := 1
	incy := 1
	cxx := []complex128{0.7 - 0.8i, -0.4 - 0.7i, -0.1 - 0.9i, 0.2 - 0.8i, -0.9 - 0.4i, 0.1 + 0.4i, -0.6 + 0.6i}
	cyy := []complex128{0.6 - 0.6i, -0.9 + 0.5i, 0.7 - 0.6i, 0.1 - 0.5i, -0.1 - 0.2i, -0.5 - 0.3i, 0.8 - 0.7i}
	c := 0.5
	s := 1.2
	cx := cvf(n)
	cy := cvf(n)
	for j := 0; j < n/7; j++ {
		for i := 0; i < 7; i++ {
			cx.Set(j*7+i, cxx[i])
			cy.Set(j*7+i, cyy[i])
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zdrot(n, cx, incx, cy, incy, c, s)
	}
}
