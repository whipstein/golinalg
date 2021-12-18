package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dlahr2 reduces the first NB columns of A real general n-BY-(n-k+1)
// matrix A so that elements below the k-th subdiagonal are zero. The
// reduction is performed by an orthogonal similarity transformation
// Q**T * A * Q. The routine returns the matrices V and T which determine
// Q as a block reflector I - V*T*V**T, and also the matrix Y = A * V * T.
//
// This is an auxiliary routine called by DGEHRD.
func Dlahr2(n, k, nb int, a *mat.Matrix, tau *mat.Vector, t, y *mat.Matrix) {
	var ei, one, zero float64
	var i int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if n <= 1 {
		return
	}

	for i = 1; i <= nb; i++ {
		if i > 1 {
			//           Update A(K+1:N,I)
			//
			//           Update I-th column of A - Y * V**T
			if err = a.Off(k, i-1).Vector().Gemv(NoTrans, n-k, i-1, -one, y.Off(k, 0), a.Off(k+i-1-1, 0).Vector(), a.Rows, one, 1); err != nil {
				panic(err)
			}

			//           Apply I - V * T**T * V**T to this column (call it b) from the
			//           left, using the last column of T as workspace
			//
			//           Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
			//                    ( V2 )             ( b2 )
			//
			//           where V1 is unit lower triangular
			//
			//           w := V1**T * b1
			t.Off(0, nb-1).Vector().Copy(i-1, a.Off(k, i-1).Vector(), 1, 1)
			if err = t.Off(0, nb-1).Vector().Trmv(Lower, Trans, Unit, i-1, a.Off(k, 0), 1); err != nil {
				panic(err)
			}

			//           w := w + V2**T * b2
			if err = t.Off(0, nb-1).Vector().Gemv(Trans, n-k-i+1, i-1, one, a.Off(k+i-1, 0), a.Off(k+i-1, i-1).Vector(), 1, one, 1); err != nil {
				panic(err)
			}

			//           w := T**T * w
			if err = t.Off(0, nb-1).Vector().Trmv(Upper, Trans, NonUnit, i-1, t, 1); err != nil {
				panic(err)
			}

			//           b2 := b2 - V2*w
			if err = a.Off(k+i-1, i-1).Vector().Gemv(NoTrans, n-k-i+1, i-1, -one, a.Off(k+i-1, 0), t.Off(0, nb-1).Vector(), 1, one, 1); err != nil {
				panic(err)
			}

			//           b1 := b1 - V1*w
			if err = t.Off(0, nb-1).Vector().Trmv(Lower, NoTrans, Unit, i-1, a.Off(k, 0), 1); err != nil {
				panic(err)
			}
			a.Off(k, i-1).Vector().Axpy(i-1, -one, t.Off(0, nb-1).Vector(), 1, 1)

			a.Set(k+i-1-1, i-1-1, ei)
		}

		//        Generate the elementary reflector H(I) to annihilate
		//        A(K+I+1:N,I)
		*a.GetPtr(k+i-1, i-1), *tau.GetPtr(i - 1) = Dlarfg(n-k-i+1, a.Get(k+i-1, i-1), a.Off(min(k+i+1, n)-1, i-1).Vector(), 1)
		ei = a.Get(k+i-1, i-1)
		a.Set(k+i-1, i-1, one)

		//        Compute  Y(K+1:N,I)
		if err = y.Off(k, i-1).Vector().Gemv(NoTrans, n-k, n-k-i+1, one, a.Off(k, i), a.Off(k+i-1, i-1).Vector(), 1, zero, 1); err != nil {
			panic(err)
		}
		if err = t.Off(0, i-1).Vector().Gemv(Trans, n-k-i+1, i-1, one, a.Off(k+i-1, 0), a.Off(k+i-1, i-1).Vector(), 1, zero, 1); err != nil {
			panic(err)
		}
		if err = y.Off(k, i-1).Vector().Gemv(NoTrans, n-k, i-1, -one, y.Off(k, 0), t.Off(0, i-1).Vector(), 1, one, 1); err != nil {
			panic(err)
		}
		y.Off(k, i-1).Vector().Scal(n-k, tau.Get(i-1), 1)

		//        Compute T(1:I,I)
		t.Off(0, i-1).Vector().Scal(i-1, -tau.Get(i-1), 1)
		err = t.Off(0, i-1).Vector().Trmv(Upper, NoTrans, NonUnit, i-1, t, 1)
		t.Set(i-1, i-1, tau.Get(i-1))

	}
	a.Set(k+nb-1, nb-1, ei)

	//     Compute Y(1:K,1:NB)
	Dlacpy(Full, k, nb, a.Off(0, 1), y)
	if err = y.Trmm(Right, Lower, NoTrans, Unit, k, nb, one, a.Off(k, 0)); err != nil {
		panic(err)
	}
	if n > k+nb {
		if err = y.Gemm(NoTrans, NoTrans, k, nb, n-k-nb, one, a.Off(0, 2+nb-1), a.Off(k+1+nb-1, 0), one); err != nil {
			panic(err)
		}
	}
	if err = y.Trmm(Right, Upper, NoTrans, NonUnit, k, nb, one, t); err != nil {
		panic(err)
	}

	return
}
