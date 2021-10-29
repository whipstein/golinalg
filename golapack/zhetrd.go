package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrd reduces a complex Hermitian matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q**H * A * Q = T.
func Zhetrd(uplo mat.MatUplo, n int, a *mat.CMatrix, d, e *mat.Vector, tau, work *mat.CVector, lwork int) (err error) {
	var lquery, upper bool
	var cone complex128
	var one float64
	var i, iws, j, kk, ldwork, lwkopt, nb, nbmin, nx int

	one = 1.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	upper = uplo == Upper
	lquery = (lwork == -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}

	if err == nil {
		//        Determine the block size.
		nb = Ilaenv(1, "Zhetrd", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = n * nb
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zhetrd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		work.Set(0, 1)
		return
	}

	nx = n
	iws = 1
	if nb > 1 && nb < n {
		//        Determine when to cross over from blocked to unblocked code
		//        (last block is always handled by unblocked code).
		nx = max(nb, Ilaenv(3, "Zhetrd", []byte{uplo.Byte()}, n, -1, -1, -1))
		if nx < n {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code by setting NX = N.
				nb = max(lwork/ldwork, 1)
				nbmin = Ilaenv(2, "Zhetrd", []byte{uplo.Byte()}, n, -1, -1, -1)
				if nb < nbmin {
					nx = n
				}
			}
		} else {
			nx = n
		}
	} else {
		nb = 1
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        Columns 1:kk are handled by the unblocked method.
		kk = n - ((n-nx+nb-1)/nb)*nb
		for i = n - nb + 1; i >= kk+1; i -= nb {
			//           Reduce columns i:i+nb-1 to tridiagonal form and form the
			//           matrix W which is needed to update the unreduced part of
			//           the matrix
			Zlatrd(uplo, i+nb-1, nb, a, e, tau, work.CMatrix(ldwork, opts))

			//           Update the unreduced submatrix A(1:i-1,1:i-1), using an
			//           update of the form:  A := A - V*W**H - W*V**H
			if err = goblas.Zher2k(uplo, NoTrans, i-1, nb, -cone, a.Off(0, i-1), work.CMatrix(ldwork, opts), one, a); err != nil {
				panic(err)
			}

			//           Copy superdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j-1-1, j-1, e.Get(j-1-1))
				d.Set(j-1, a.GetRe(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		if err = Zhetd2(uplo, kk, a, d, e, tau); err != nil {
			panic(err)
		}
	} else {
		//        Reduce the lower triangle of A
		for i = 1; i <= n-nx; i += nb {
			//           Reduce columns i:i+nb-1 to tridiagonal form and form the
			//           matrix W which is needed to update the unreduced part of
			//           the matrix
			Zlatrd(uplo, n-i+1, nb, a.Off(i-1, i-1), e.Off(i-1), tau.Off(i-1), work.CMatrix(ldwork, opts))

			//           Update the unreduced submatrix A(i+nb:n,i+nb:n), using
			//           an update of the form:  A := A - V*W**H - W*V**H
			if err = goblas.Zher2k(uplo, NoTrans, n-i-nb+1, nb, -cone, a.Off(i+nb-1, i-1), work.CMatrixOff(nb, ldwork, opts), one, a.Off(i+nb-1, i+nb-1)); err != nil {
				panic(err)
			}

			//           Copy subdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j, j-1, e.Get(j-1))
				d.Set(j-1, a.GetRe(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		if err = Zhetd2(uplo, n-i+1, a.Off(i-1, i-1), d.Off(i-1), e.Off(i-1), tau.Off(i-1)); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(lwkopt))

	return
}
