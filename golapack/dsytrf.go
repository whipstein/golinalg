package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dsytrf computes the factorization of a real symmetric matrix A using
// the Bunch-Kaufman diagonal pivoting method.  The form of the
// factorization is
//
//    A = U**T*D*U  or  A = L*D*L**T
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is symmetric and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Dsytrf(uplo byte, n *int, a *mat.Matrix, lda *int, ipiv *[]int, work *mat.Vector, lwork, info *int) {
	var lquery, upper bool
	var iinfo, iws, j, k, kb, ldwork, lwkopt, nb, nbmin int

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
		(*info) = -7
	}

	if (*info) == 0 {
		//        Determine the block size
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRF"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		lwkopt = (*n) * nb
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRF"), -(*info))
		return
	} else if lquery {
		return
	}

	nbmin = 2
	ldwork = (*n)
	if nb > 1 && nb < (*n) {
		iws = ldwork * nb
		if (*lwork) < iws {
			nb = maxint((*lwork)/ldwork, 1)
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("DSYTRF"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
		}
	} else {
		iws = 1
	}
	if nb < nbmin {
		nb = (*n)
	}

	if upper {
		//        Factorize A as U**T*D*U using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        KB, where KB is the number of columns factorized by DLASYF;
		//        KB is either NB or NB-1, or K for the last block
		k = (*n)
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			goto label40
		}

		if k > nb {
			//           Factorize columns k-kb+1:k of A and use blocked code to
			//           update columns 1:k-kb
			Dlasyf(uplo, &k, &nb, &kb, a, lda, ipiv, work.Matrix(ldwork, opts), &ldwork, &iinfo)
		} else {
			//           Use unblocked code to factorize columns 1:k of A
			Dsytf2(uplo, &k, a, lda, ipiv, &iinfo)
			kb = k
		}

		//        Set INFO on the first occurrence of a zero pivot
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo
		}

		//        Decrease K and return to the start of the main loop
		k = k - kb
		goto label10

	} else {
		//        Factorize A as L*D*L**T using the lower triangle of A
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        KB, where KB is the number of columns factorized by DLASYF;
		//        KB is either NB or NB-1, or N-K+1 for the last block
		k = 1
	label20:
		;

		//        If K > N, exit from loop
		if k > (*n) {
			goto label40
		}

		if k <= (*n)-nb {
			//           Factorize columns k:k+kb-1 of A and use blocked code to
			//           update columns k+kb:n
			Dlasyf(uplo, toPtr((*n)-k+1), &nb, &kb, a.Off(k-1, k-1), lda, toSlice(ipiv, k-1), work.Matrix(ldwork, opts), &ldwork, &iinfo)
		} else {
			//           Use unblocked code to factorize columns k:n of A
			_ipiv := (*ipiv)[k-1:]
			Dsytf2(uplo, toPtr((*n)-k+1), a.Off(k-1, k-1), lda, &_ipiv, &iinfo)
			kb = (*n) - k + 1
		}

		//        Set INFO on the first occurrence of a zero pivot
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo + k - 1
		}

		//        Adjust IPIV
		for j = k; j <= k+kb-1; j++ {
			if (*ipiv)[j-1] > 0 {
				(*ipiv)[j-1] = (*ipiv)[j-1] + k - 1
			} else {
				(*ipiv)[j-1] = (*ipiv)[j-1] - k + 1
			}
		}

		//        Increase K and return to the start of the main loop
		k = k + kb
		goto label20

	}

label40:
	;
	work.Set(0, float64(lwkopt))
}
