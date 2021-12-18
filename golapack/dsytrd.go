package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytrd reduces a real symmetric matrix A to real symmetric
// tridiagonal form T by an orthogonal similarity transformation:
// Q**T * A * Q = T.
func Dsytrd(uplo mat.MatUplo, n int, a *mat.Matrix, d, e, tau, work *mat.Vector, lwork int) (err error) {
	var lquery, upper bool
	var one float64
	var i, iws, j, kk, ldwork, lwkopt, nb, nbmin, nx int

	one = 1.0

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
		nb = Ilaenv(1, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = n * nb
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dsytrd", err)
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
		nx = max(nb, Ilaenv(3, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1))
		if nx < n {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code by setting NX = N.
				nb = max(lwork/ldwork, 1)
				nbmin = Ilaenv(2, "Dsytrd", []byte{uplo.Byte()}, n, -1, -1, -1)
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
			Dlatrd(uplo, i+nb-1, nb, a, e, tau, work.Matrix(ldwork, opts))

			//           Update the unreduced submatrix A(1:i-1,1:i-1), using an
			//           update of the form:  A := A - V*W**T - W*V**T
			if err = a.Syr2k(uplo, NoTrans, i-1, nb, -one, a.Off(0, i-1), work.Matrix(ldwork, opts), one); err != nil {
				panic(err)
			}

			//           Copy superdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.Set(j-1-1, j-1, e.Get(j-1-1))
				d.Set(j-1, a.Get(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		if err = Dsytd2(uplo, kk, a, d, e, tau); err != nil {
			panic(err)
		}
	} else {
		//        Reduce the lower triangle of A
		for i = 1; i <= n-nx; i += nb {
			//           Reduce columns i:i+nb-1 to tridiagonal form and form the
			//           matrix W which is needed to update the unreduced part of
			//           the matrix
			Dlatrd(uplo, n-i+1, nb, a.Off(i-1, i-1), e.Off(i-1), tau.Off(i-1), work.Matrix(ldwork, opts))

			//           Update the unreduced submatrix A(i+ib:n,i+ib:n), using
			//           an update of the form:  A := A - V*W**T - W*V**T
			err = a.Off(i+nb-1, i+nb-1).Syr2k(uplo, NoTrans, n-i-nb+1, nb, -one, a.Off(i+nb-1, i-1), work.Off(nb).Matrix(ldwork, opts), one)

			//           Copy subdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.Set(j, j-1, e.Get(j-1))
				d.Set(j-1, a.Get(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		if err = Dsytd2(uplo, n-i+1, a.Off(i-1, i-1), d.Off(i-1), e.Off(i-1), tau.Off(i-1)); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(lwkopt))

	return
}
