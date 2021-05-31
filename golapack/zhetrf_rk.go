package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrfrk computes the factorization of a complex Hermitian matrix A
// using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
//
//    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**H (or L**H) is the conjugate of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is Hermitian and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
// For more information see Further Details section.
func Zhetrfrk(uplo byte, n *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var i, iinfo, ip, iws, k, kb, ldwork, lwkopt, nb, nbmin int

	//     Test the input parameters.
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
		(*info) = -8
	}

	if (*info) == 0 {
		//        Determine the block size
		nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRF_RK"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))
		lwkopt = (*n) * nb
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRF_RK"), -(*info))
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
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZHETRF_RK"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
		}
	} else {
		iws = 1
	}
	if nb < nbmin {
		nb = (*n)
	}

	if upper {
		//        Factorize A as U*D*U**T using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        KB, where KB is the number of columns factorized by ZLAHEF_RK;
		//        KB is either NB or NB-1, or K for the last block
		k = (*n)
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			goto label15
		}

		if k > nb {
			//           Factorize columns k-kb+1:k of A and use blocked code to
			//           update columns 1:k-kb
			Zlahefrk(uplo, &k, &nb, &kb, a, lda, e, ipiv, work.CMatrix(ldwork, opts), &ldwork, &iinfo)
		} else {
			//           Use unblocked code to factorize columns 1:k of A
			Zhetf2rk(uplo, &k, a, lda, e, ipiv, &iinfo)
			kb = k
		}

		//        Set INFO on the first occurrence of a zero pivot
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo
		}

		//        No need to adjust IPIV
		//
		//
		//        Apply permutations to the leading panel 1:k-1
		//
		//        Read IPIV from the last block factored, i.e.
		//        indices  k-kb+1:k and apply row permutations to the
		//        last k+1 colunms k+1:N after that block
		//        (We can do the simple loop over IPIV with decrement -1,
		//        since the ABS value of IPIV( I ) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		if k < (*n) {
			for i = k; i >= (k - kb + 1); i-- {
				ip = absint((*ipiv)[i-1])
				if ip != i {
					goblas.Zswap(toPtr((*n)-k), a.CVector(i-1, k+1-1), lda, a.CVector(ip-1, k+1-1), lda)
				}
			}
		}

		//        Decrease K and return to the start of the main loop
		k = k - kb
		goto label10

		//        This label is the exit from main loop over K decreasing
		//        from N to 1 in steps of KB
	label15:
	} else {
		//        Factorize A as L*D*L**T using the lower triangle of A
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        KB, where KB is the number of columns factorized by ZLAHEF_RK;
		//        KB is either NB or NB-1, or N-K+1 for the last block
		k = 1
	label20:
		;

		//        If K > N, exit from loop
		if k > (*n) {
			goto label35
		}

		if k <= (*n)-nb {
			//           Factorize columns k:k+kb-1 of A and use blocked code to
			//           update columns k+kb:n
			Zlahefrk(uplo, toPtr((*n)-k+1), &nb, &kb, a.Off(k-1, k-1), lda, e.Off(k-1), toSlice(ipiv, k-1), work.CMatrix(ldwork, opts), &ldwork, &iinfo)
		} else {
			//           Use unblocked code to factorize columns k:n of A
			Zhetf2rk(uplo, toPtr((*n)-k+1), a.Off(k-1, k-1), lda, e.Off(k-1), toSlice(ipiv, k-1), &iinfo)
			kb = (*n) - k + 1

		}

		//        Set INFO on the first occurrence of a zero pivot
		if (*info) == 0 && iinfo > 0 {
			(*info) = iinfo + k - 1
		}

		//        Adjust IPIV
		for i = k; i <= k+kb-1; i++ {
			if (*ipiv)[i-1] > 0 {
				(*ipiv)[i-1] = (*ipiv)[i-1] + k - 1
			} else {
				(*ipiv)[i-1] = (*ipiv)[i-1] - k + 1
			}
		}

		//        Apply permutations to the leading panel 1:k-1
		//
		//        Read IPIV from the last block factored, i.e.
		//        indices  k:k+kb-1 and apply row permutations to the
		//        first k-1 colunms 1:k-1 before that block
		//        (We can do the simple loop over IPIV with increment 1,
		//        since the ABS value of IPIV( I ) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		if k > 1 {
			for i = k; i <= (k + kb - 1); i++ {
				ip = absint((*ipiv)[i-1])
				if ip != i {
					goblas.Zswap(toPtr(k-1), a.CVector(i-1, 0), lda, a.CVector(ip-1, 0), lda)
				}
			}
		}

		//        Increase K and return to the start of the main loop
		k = k + kb
		goto label20

		//        This label is the exit from main loop over K increasing
		//        from 1 to N in steps of KB
	label35:

		//     End Lower
	}

	work.SetRe(0, float64(lwkopt))
}
