package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgbtrs solves a system of linear equations
//    A * X = B  or  A**T * X = B
// with a general band matrix A using the LU factorization computed
// by DGBTRF.
func Dgbtrs(trans mat.MatTrans, n, kl, ku, nrhs int, ab *mat.Matrix, ipiv []int, b *mat.Matrix) (err error) {
	var lnoti, notran bool
	var one float64
	var i, j, kd, l, lm int

	one = 1.0

	//     Test the input parameters.
	notran = trans == NoTrans
	if !trans.IsValid() {
		err = fmt.Errorf("!trans.IsValid(): trans=%s", trans)
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
		gltest.Xerbla2("Dgbtrs", err)
		return err
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return err
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
				l = ipiv[j-1]
				if l != j {
					b.Off(j-1, 0).Vector().Swap(nrhs, b.Off(l-1, 0).Vector(), b.Rows, b.Rows)
				}
				err = b.Off(j, 0).Ger(lm, nrhs, -one, ab.Off(kd, j-1).Vector(), 1, b.Off(j-1, 0).Vector(), b.Rows)
			}
		}

		for i = 1; i <= nrhs; i++ {
			//           Solve U*X = B, overwriting B with X.
			err = b.Off(0, i-1).Vector().Tbsv(Upper, NoTrans, NonUnit, n, kl+ku, ab, 1)
		}

	} else {
		//        Solve A**T*X = B.
		for i = 1; i <= nrhs; i++ {
			//           Solve U**T*X = B, overwriting B with X.
			err = b.Off(0, i-1).Vector().Tbsv(Upper, Trans, NonUnit, n, kl+ku, ab, 1)
		}

		//        Solve L**T*X = B, overwriting B with X.
		if lnoti {
			for j = n - 1; j >= 1; j-- {
				lm = min(kl, n-j)
				err = b.Off(j-1, 0).Vector().Gemv(Trans, lm, nrhs, -one, b.Off(j, 0), ab.Off(kd, j-1).Vector(), 1, one, b.Rows)
				l = ipiv[j-1]
				if l != j {
					b.Off(j-1, 0).Vector().Swap(nrhs, b.Off(l-1, 0).Vector(), b.Rows, b.Rows)
				}
			}
		}
	}

	return err
}
