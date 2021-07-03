package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrdhe2hb reduces a complex Hermitian matrix A to complex Hermitian
// band-diagonal form AB by a unitary similarity transformation:
// Q**H * A * Q = AB.
func Zhetrdhe2hb(uplo byte, n, kd *int, a *mat.CMatrix, lda *int, ab *mat.CMatrix, ldab *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var half, one, zero complex128
	var rone float64
	var i, iinfo, j, lds1, lds2, ldt, ldw, lk, ls1, ls2, lt, lw, lwmin, pk, pn, s1pos, s2pos, tpos, wpos int
	var err error
	_ = err

	rone = 1.0
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)

	//     Determine the minimal workspace size required
	//     and test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	lwmin = Ilaenv2stage(func() *int { y := 4; return &y }(), []byte("ZHETRD_HE2HB"), []byte{' '}, n, kd, toPtr(-1), toPtr(-1))
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*kd) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldab) < maxint(1, (*kd)+1) {
		(*info) = -7
	} else if (*lwork) < lwmin && !lquery {
		(*info) = -10
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRD_HE2HB"), -(*info))
		return
	} else if lquery {
		work.SetRe(0, float64(lwmin))
		return
	}

	//     Quick return if possible
	//     Copy the upper/lower portion of A into AB
	if (*n) <= (*kd)+1 {
		if upper {
			for i = 1; i <= (*n); i++ {
				lk = minint((*kd)+1, i)
				goblas.Zcopy(lk, a.CVector(i-lk+1-1, i-1), 1, ab.CVector((*kd)+1-lk+1-1, i-1), 1)
			}
		} else {
			for i = 1; i <= (*n); i++ {
				lk = minint((*kd)+1, (*n)-i+1)
				goblas.Zcopy(lk, a.CVector(i-1, i-1), 1, ab.CVector(0, i-1), 1)
			}
		}
		work.Set(0, 1)
		return
	}

	//     Determine the pointer position for the workspace
	ldt = (*kd)
	lds1 = (*kd)
	lt = ldt * (*kd)
	lw = (*n) * (*kd)
	ls1 = lds1 * (*kd)
	ls2 = lwmin - lt - lw - ls1
	//      LS2 = N*maxint(KD,FACTOPTNB)
	tpos = 1
	wpos = tpos + lt
	s1pos = wpos + lw
	s2pos = s1pos + ls1
	if upper {
		ldw = (*kd)
		lds2 = (*kd)
	} else {
		ldw = (*n)
		lds2 = (*n)
	}

	//     Set the workspace of the triangular matrix T to zero once such a
	//     way every time T is generated the upper/lower portion will be always zero
	Zlaset('A', &ldt, kd, &zero, &zero, work.CMatrixOff(tpos-1, ldt, opts), &ldt)

	if upper {
		for i = 1; i <= (*n)-(*kd); i += (*kd) {
			pn = (*n) - i - (*kd) + 1
			pk = minint((*n)-i-(*kd)+1, *kd)

			//            Compute the LQ factorization of the current block
			Zgelqf(kd, &pn, a.Off(i-1, i+(*kd)-1), lda, tau.Off(i-1), work.Off(s2pos-1), &ls2, &iinfo)

			//            Copy the upper portion of A into AB
			for j = i; j <= i+pk-1; j++ {
				lk = minint(*kd, (*n)-j) + 1
				goblas.Zcopy(lk, a.CVector(j-1, j-1), *lda, ab.CVector((*kd)+1-1, j-1), (*ldab)-1)
			}

			Zlaset('L', &pk, &pk, &zero, &one, a.Off(i-1, i+(*kd)-1), lda)

			//            Form the matrix T
			Zlarft('F', 'R', &pn, &pk, a.Off(i-1, i+(*kd)-1), lda, tau.Off(i-1), work.CMatrixOff(tpos-1, ldt, opts), &ldt)

			//            Compute W:
			err = goblas.Zgemm(ConjTrans, NoTrans, pk, pn, pk, one, work.CMatrixOff(tpos-1, ldt, opts), ldt, a.Off(i-1, i+(*kd)-1), *lda, zero, work.CMatrixOff(s2pos-1, lds2, opts), lds2)

			err = goblas.Zhemm(Right, mat.UploByte(uplo), pk, pn, one, a.Off(i+(*kd)-1, i+(*kd)-1), *lda, work.CMatrixOff(s2pos-1, lds2, opts), lds2, zero, work.CMatrixOff(wpos-1, ldw, opts), ldw)

			err = goblas.Zgemm(NoTrans, ConjTrans, pk, pk, pn, one, work.CMatrixOff(wpos-1, ldw, opts), ldw, work.CMatrixOff(s2pos-1, lds2, opts), lds2, zero, work.CMatrixOff(s1pos-1, lds1, opts), lds1)

			err = goblas.Zgemm(NoTrans, NoTrans, pk, pn, pk, -half, work.CMatrixOff(s1pos-1, lds1, opts), lds1, a.Off(i-1, i+(*kd)-1), *lda, one, work.CMatrixOff(wpos-1, ldw, opts), ldw)

			//            Update the unreduced submatrix A(i+kd:n,i+kd:n), using
			//            an update of the form:  A := A - V'*W - W'*V
			err = goblas.Zher2k(mat.UploByte(uplo), ConjTrans, pn, pk, -one, a.Off(i-1, i+(*kd)-1), *lda, work.CMatrixOff(wpos-1, ldw, opts), ldw, rone, a.Off(i+(*kd)-1, i+(*kd)-1), *lda)
		}

		//        Copy the upper band to AB which is the band storage matrix
		for j = (*n) - (*kd) + 1; j <= (*n); j++ {
			lk = minint(*kd, (*n)-j) + 1
			goblas.Zcopy(lk, a.CVector(j-1, j-1), *lda, ab.CVector((*kd)+1-1, j-1), (*ldab)-1)
		}

	} else {
		//         Reduce the lower triangle of A to lower band matrix
		for i = 1; i <= (*n)-(*kd); i += (*kd) {
			pn = (*n) - i - (*kd) + 1
			pk = minint((*n)-i-(*kd)+1, *kd)

			//            Compute the QR factorization of the current block
			Zgeqrf(&pn, kd, a.Off(i+(*kd)-1, i-1), lda, tau.Off(i-1), work.Off(s2pos-1), &ls2, &iinfo)

			//            Copy the upper portion of A into AB
			for j = i; j <= i+pk-1; j++ {
				lk = minint(*kd, (*n)-j) + 1
				goblas.Zcopy(lk, a.CVector(j-1, j-1), 1, ab.CVector(0, j-1), 1)
			}

			Zlaset('U', &pk, &pk, &zero, &one, a.Off(i+(*kd)-1, i-1), lda)

			//            Form the matrix T
			Zlarft('F', 'C', &pn, &pk, a.Off(i+(*kd)-1, i-1), lda, tau.Off(i-1), work.CMatrixOff(tpos-1, ldt, opts), &ldt)

			//            Compute W:
			err = goblas.Zgemm(NoTrans, NoTrans, pn, pk, pk, one, a.Off(i+(*kd)-1, i-1), *lda, work.CMatrixOff(tpos-1, ldt, opts), ldt, zero, work.CMatrixOff(s2pos-1, lds2, opts), lds2)

			err = goblas.Zhemm(Left, mat.UploByte(uplo), pn, pk, one, a.Off(i+(*kd)-1, i+(*kd)-1), *lda, work.CMatrixOff(s2pos-1, lds2, opts), lds2, zero, work.CMatrixOff(wpos-1, ldw, opts), ldw)

			err = goblas.Zgemm(ConjTrans, NoTrans, pk, pk, pn, one, work.CMatrixOff(s2pos-1, lds2, opts), lds2, work.CMatrixOff(wpos-1, ldw, opts), ldw, zero, work.CMatrixOff(s1pos-1, lds1, opts), lds1)

			err = goblas.Zgemm(NoTrans, NoTrans, pn, pk, pk, -half, a.Off(i+(*kd)-1, i-1), *lda, work.CMatrixOff(s1pos-1, lds1, opts), lds1, one, work.CMatrixOff(wpos-1, ldw, opts), ldw)

			//            Update the unreduced submatrix A(i+kd:n,i+kd:n), using
			//            an update of the form:  A := A - V*W' - W*V'
			err = goblas.Zher2k(mat.UploByte(uplo), NoTrans, pn, pk, -one, a.Off(i+(*kd)-1, i-1), *lda, work.CMatrixOff(wpos-1, ldw, opts), ldw, rone, a.Off(i+(*kd)-1, i+(*kd)-1), *lda)
			//            ==================================================================
			//            RESTORE A FOR COMPARISON AND CHECKING TO BE REMOVED
			//             DO 45 J = I, I+PK-1
			//                LK = minint( KD, N-J ) + 1
			//                CALL ZCOPY( LK, AB( 1, J ), 1, A( J, J ), 1 )
			//   45        CONTINUE
			//            ==================================================================
		}

		//        Copy the lower band to AB which is the band storage matrix
		for j = (*n) - (*kd) + 1; j <= (*n); j++ {
			lk = minint(*kd, (*n)-j) + 1
			goblas.Zcopy(lk, a.CVector(j-1, j-1), 1, ab.CVector(0, j-1), 1)
		}
	}

	work.SetRe(0, float64(lwmin))
}
