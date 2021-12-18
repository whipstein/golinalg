package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrfrk computes the factorization of a complex symmetric matrix A
// using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
// For more information see Further Details section.
func ZsytrfRk(uplo mat.MatUplo, n int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, upper bool
	var i, iinfo, ip, iws, k, kb, ldwork, lwkopt, nb, nbmin int

	//     Test the input parameters.
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
		//        Determine the block size
		nb = Ilaenv(1, "ZsytrfRk", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = n * nb
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("ZsytrfRk", err)
		return
	} else if lquery {
		return
	}

	nbmin = 2
	ldwork = n
	if nb > 1 && nb < n {
		iws = ldwork * nb
		if lwork < iws {
			nb = max(lwork/ldwork, 1)
			nbmin = max(2, Ilaenv(2, "ZsytrfRk", []byte{uplo.Byte()}, n, -1, -1, -1))
		}
	} else {
		iws = 1
	}
	if nb < nbmin {
		nb = n
	}

	if upper {
		//        Factorize A as U*D*U**T using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        KB, where KB is the number of columns factorized by ZLASYF_RK;
		//        KB is either NB or NB-1, or K for the last block
		k = n
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			goto label15
		}

		if k > nb {
			//           Factorize columns k-kb+1:k of A and use blocked code to
			//           update columns 1:k-kb
			kb, iinfo = ZlasyfRk(uplo, k, nb, a, e, ipiv, work.CMatrix(ldwork, opts))
		} else {
			//           Use unblocked code to factorize columns 1:k of A
			if iinfo, err = Zsytf2Rk(uplo, k, a, e, ipiv); err != nil {
				panic(err)
			}
			kb = k
		}

		//        Set INFO on the first occurrence of a zero pivot
		if info == 0 && iinfo > 0 {
			info = iinfo
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
		if k < n {
			for i = k; i >= (k - kb + 1); i-- {
				ip = abs((*ipiv)[i-1])
				if ip != i {
					a.Off(ip-1, k).CVector().Swap(n-k, a.Off(i-1, k).CVector(), a.Rows, a.Rows)
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
		//        KB, where KB is the number of columns factorized by ZLASYF_RK;
		//        KB is either NB or NB-1, or N-K+1 for the last block
		k = 1
	label20:
		;

		//        If K > N, exit from loop
		if k > n {
			goto label35
		}

		if k <= n-nb {
			//           Factorize columns k:k+kb-1 of A and use blocked code to
			//           update columns k+kb:n
			kb, iinfo = ZlasyfRk(uplo, n-k+1, nb, a.Off(k-1, k-1), e.Off(k-1), toSlice(ipiv, k-1), work.CMatrix(ldwork, opts))
		} else {
			//           Use unblocked code to factorize columns k:n of A
			if iinfo, err = Zsytf2Rk(uplo, n-k+1, a.Off(k-1, k-1), e.Off(k-1), toSlice(ipiv, k-1)); err != nil {
				panic(err)
			}
			kb = n - k + 1

		}

		//        Set INFO on the first occurrence of a zero pivot
		if info == 0 && iinfo > 0 {
			info = iinfo + k - 1
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
				ip = abs((*ipiv)[i-1])
				if ip != i {
					a.Off(ip-1, 0).CVector().Swap(k-1, a.Off(i-1, 0).CVector(), a.Rows, a.Rows)
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

	return
}
