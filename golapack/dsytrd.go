package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytrd reduces a real symmetric matrix A to real symmetric
// tridiagonal form T by an orthogonal similarity transformation:
// Q**T * A * Q = T.
func Dsytrd(uplo byte, n *int, a *mat.Matrix, lda *int, d, e, tau, work *mat.Vector, lwork, info *int) {
	var lquery, upper bool
	var one float64
	var i, iinfo, iws, j, kk, ldwork, lwkopt, nb, nbmin, nx int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*lwork) < 1 && !lquery {
		(*info) = -9
	}

	if (*info) == 0 {
		//        Determine the block size.
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		lwkopt = (*n) * nb
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		work.Set(0, 1)
		return
	}

	nx = (*n)
	iws = 1
	if nb > 1 && nb < (*n) {
		//        Determine when to cross over from blocked to unblocked code
		//        (last block is always handled by unblocked code).
		nx = maxint(nb, Ilaenv(func() *int { y := 3; return &y }(), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
		if nx < (*n) {
			//           Determine if workspace is large enough for blocked code.
			ldwork = (*n)
			iws = ldwork * nb
			if (*lwork) < iws {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code by setting NX = N.
				nb = maxint((*lwork)/ldwork, 1)
				nbmin = Ilaenv(func() *int { y := 2; return &y }(), []byte("DSYTRD"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
				if nb < nbmin {
					nx = (*n)
				}
			}
		} else {
			nx = (*n)
		}
	} else {
		nb = 1
	}

	if upper {
		//        Reduce the upper triangle of A.
		//        Columns 1:kk are handled by the unblocked method.
		kk = (*n) - (((*n)-nx+nb-1)/nb)*nb
		for i = (*n) - nb + 1; i >= kk+1; i -= nb {
			//           Reduce columns i:i+nb-1 to tridiagonal form and form the
			//           matrix W which is needed to update the unreduced part of
			//           the matrix
			Dlatrd(uplo, toPtr(i+nb-1), &nb, a, lda, e, tau, work.Matrix(ldwork, opts), &ldwork)

			//           Update the unreduced submatrix A(1:i-1,1:i-1), using an
			//           update of the form:  A := A - V*W**T - W*V**T
			err = goblas.Dsyr2k(mat.UploByte(uplo), NoTrans, i-1, nb, -one, a.Off(0, i-1), *lda, work.Matrix(ldwork, opts), ldwork, one, a, *lda)

			//           Copy superdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.Set(j-1-1, j-1, e.Get(j-1-1))
				d.Set(j-1, a.Get(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		Dsytd2(uplo, &kk, a, lda, d, e, tau, &iinfo)
	} else {
		//        Reduce the lower triangle of A
		for i = 1; i <= (*n)-nx; i += nb {
			//           Reduce columns i:i+nb-1 to tridiagonal form and form the
			//           matrix W which is needed to update the unreduced part of
			//           the matrix
			Dlatrd(uplo, toPtr((*n)-i+1), &nb, a.Off(i-1, i-1), lda, e.Off(i-1), tau.Off(i-1), work.Matrix(ldwork, opts), &ldwork)

			//           Update the unreduced submatrix A(i+ib:n,i+ib:n), using
			//           an update of the form:  A := A - V*W**T - W*V**T
			err = goblas.Dsyr2k(mat.UploByte(uplo), NoTrans, (*n)-i-nb+1, nb, -one, a.Off(i+nb-1, i-1), *lda, work.MatrixOff(nb+1-1, ldwork, opts), ldwork, one, a.Off(i+nb-1, i+nb-1), *lda)

			//           Copy subdiagonal elements back into A, and diagonal
			//           elements into D
			for j = i; j <= i+nb-1; j++ {
				a.Set(j+1-1, j-1, e.Get(j-1))
				d.Set(j-1, a.Get(j-1, j-1))
			}
		}

		//        Use unblocked code to reduce the last or only block
		Dsytd2(uplo, toPtr((*n)-i+1), a.Off(i-1, i-1), lda, d.Off(i-1), e.Off(i-1), tau.Off(i-1), &iinfo)
	}

	work.Set(0, float64(lwkopt))
}
