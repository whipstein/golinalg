package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgehrd reduces a complex general matrix A to upper Hessenberg form H by
// an unitary similarity transformation:  Q**H * A * Q = H .
func Zgehrd(n, ilo, ihi *int, a *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var ei, one, zero complex128
	var i, ib, iinfo, iwt, j, ldt, ldwork, lwkopt, nb, nbmax, nbmin, nh, nx, tsize int
	opts := mat.NewMatOptsCol()

	nbmax = 64
	ldt = nbmax + 1
	tsize = ldt * nbmax
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*ilo) < 1 || (*ilo) > maxint(1, *n) {
		(*info) = -2
	} else if (*ihi) < minint(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -8
	}

	if (*info) == 0 {
		//        Compute the workspace requirements
		nb = minint(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEHRD"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
		lwkopt = (*n)*nb + tsize
		work.SetRe(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGEHRD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Set elements 1:ILO-1 and IHI:N-1 of TAU to zero
	for i = 1; i <= (*ilo)-1; i++ {
		tau.Set(i-1, zero)
	}
	for i = maxint(1, *ihi); i <= (*n)-1; i++ {
		tau.Set(i-1, zero)
	}

	//     Quick return if possible
	nh = (*ihi) - (*ilo) + 1
	if nh <= 1 {
		work.Set(0, 1)
		return
	}

	//     Determine the block size
	nb = minint(nbmax, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEHRD"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
	nbmin = 2
	if nb > 1 && nb < nh {
		//        Determine when to cross over from blocked to unblocked code
		//        (last block is always handled by unblocked code)
		nx = maxint(nb, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGEHRD"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
		if nx < nh {
			//           Determine if workspace is large enough for blocked code
			if (*lwork) < (*n)*nb+tsize {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGEHRD"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
				if (*lwork) >= ((*n)*nbmin + tsize) {
					nb = ((*lwork) - tsize) / (*n)
				} else {
					nb = 1
				}
			}
		}
	}
	ldwork = (*n)
	//
	if nb < nbmin || nb >= nh {
		//        Use unblocked code below
		i = (*ilo)

	} else {
		//        Use blocked code
		iwt = 1 + (*n)*nb
		for i = (*ilo); i <= (*ihi)-1-nx; i += nb {
			ib = minint(nb, (*ihi)-i)

			//           Reduce columns i:i+ib-1 to Hessenberg form, returning the
			//           matrices V and T of the block reflector H = I - V*T*V**H
			//           which performs the reduction, and also the matrix Y = A*V*T
			Zlahr2(ihi, &i, &ib, a.Off(0, i-1), lda, tau.Off(i-1), work.CMatrixOff(iwt-1, ldt, opts), &ldt, work.CMatrix(ldwork, opts), &ldwork)

			//           Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
			//           right, computing  A := A - Y * V**H. V(i+ib,ib-1) must be set
			//           to 1
			ei = a.Get(i+ib-1, i+ib-1-1)
			a.Set(i+ib-1, i+ib-1-1, one)
			goblas.Zgemm(NoTrans, ConjTrans, ihi, toPtr((*ihi)-i-ib+1), &ib, toPtrc128(-one), work.CMatrix(ldwork, opts), &ldwork, a.Off(i+ib-1, i-1), lda, &one, a.Off(0, i+ib-1), lda)
			a.Set(i+ib-1, i+ib-1-1, ei)

			//           Apply the block reflector H to A(1:i,i+1:i+ib-1) from the
			//           right
			goblas.Ztrmm(Right, Lower, ConjTrans, Unit, &i, toPtr(ib-1), &one, a.Off(i+1-1, i-1), lda, work.CMatrix(ldwork, opts), &ldwork)
			for j = 0; j <= ib-2; j++ {
				goblas.Zaxpy(&i, toPtrc128(-one), work.Off(ldwork*j+1-1), func() *int { y := 1; return &y }(), a.CVector(0, i+j+1-1), func() *int { y := 1; return &y }())
			}

			//           Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
			//           left
			Zlarfb('L', 'C', 'F', 'C', toPtr((*ihi)-i), toPtr((*n)-i-ib+1), &ib, a.Off(i+1-1, i-1), lda, work.CMatrixOff(iwt-1, ldt, opts), &ldt, a.Off(i+1-1, i+ib-1), lda, work.CMatrix(ldwork, opts), &ldwork)
		}
	}

	//     Use unblocked code to reduce the rest of the matrix
	Zgehd2(n, &i, ihi, a, lda, tau, work, &iinfo)
	work.SetRe(0, float64(lwkopt))
}
