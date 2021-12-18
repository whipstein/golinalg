package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbtrs solves a system of linear equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B
// with a general band matrix A using the LU factorization computed
// by ZGBTRF.
func Zgbtrs(trans mat.MatTrans, n, kl, ku, nrhs int, ab *mat.CMatrix, ipiv *[]int, b *mat.CMatrix) (err error) {
	var lnoti, notran bool
	var one complex128
	var i, j, kd, l, lm int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	notran = trans == NoTrans
	if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < (2*kl + ku + 1) {
		err = fmt.Errorf("ab.Rows < (2*kl + ku + 1): ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgbtrs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	kd = ku + kl + 1
	lnoti = kl > 0

	if notran {
		//        Solve  A*X = B.
		//
		//        Solve L*X = B, overwriting B with X.
		//
		//        L is represented as a product of permutations and unit lower
		//        triangular matrices L = P(1) * L(1) * ... * P(n-1) * L(n-1),
		//        where each transformation L(i) is a rank-one modification of
		//        the identity matrix.
		if lnoti {
			for j = 1; j <= n-1; j++ {
				lm = min(kl, n-j)
				l = (*ipiv)[j-1]
				if l != j {
					b.Off(j-1, 0).CVector().Swap(nrhs, b.Off(l-1, 0).CVector(), b.Rows, b.Rows)
				}
				if err = b.Off(j, 0).Geru(lm, nrhs, -one, ab.Off(kd, j-1).CVector(), 1, b.Off(j-1, 0).CVector(), b.Rows); err != nil {
					panic(err)
				}
			}
		}

		for i = 1; i <= nrhs; i++ {
			//           Solve U*X = B, overwriting B with X.
			if err = b.Off(0, i-1).CVector().Tbsv(Upper, NoTrans, NonUnit, n, kl+ku, ab, 1); err != nil {
				panic(err)
			}
		}

	} else if trans == Trans {
		//        Solve A**T * X = B.
		for i = 1; i <= nrhs; i++ {
			//           Solve U**T * X = B, overwriting B with X.
			if err = b.Off(0, i-1).CVector().Tbsv(Upper, Trans, NonUnit, n, kl+ku, ab, 1); err != nil {
				panic(err)
			}
		}

		//        Solve L**T * X = B, overwriting B with X.
		if lnoti {
			for j = n - 1; j >= 1; j-- {
				lm = min(kl, n-j)
				if err = b.Off(j-1, 0).CVector().Gemv(Trans, lm, nrhs, -one, b.Off(j, 0), ab.Off(kd, j-1).CVector(), 1, one, b.Rows); err != nil {
					panic(err)
				}
				l = (*ipiv)[j-1]
				if l != j {
					b.Off(j-1, 0).CVector().Swap(nrhs, b.Off(l-1, 0).CVector(), b.Rows, b.Rows)
				}
			}
		}

	} else {
		//        Solve A**H * X = B.
		for i = 1; i <= nrhs; i++ {
			//           Solve U**H * X = B, overwriting B with X.
			if err = b.Off(0, i-1).CVector().Tbsv(Upper, ConjTrans, NonUnit, n, kl+ku, ab, 1); err != nil {
				panic(err)
			}
		}

		//        Solve L**H * X = B, overwriting B with X.
		if lnoti {
			for j = n - 1; j >= 1; j-- {
				lm = min(kl, n-j)
				Zlacgv(nrhs, b.Off(j-1, 0).CVector(), b.Rows)
				err = b.Off(j-1, 0).CVector().Gemv(ConjTrans, lm, nrhs, -one, b.Off(j, 0), ab.Off(kd, j-1).CVector(), 1, one, b.Rows)
				Zlacgv(nrhs, b.Off(j-1, 0).CVector(), b.Rows)
				l = (*ipiv)[j-1]
				if l != j {
					b.Off(j-1, 0).CVector().Swap(nrhs, b.Off(l-1, 0).CVector(), b.Rows, b.Rows)
				}
			}
		}
	}

	return
}
