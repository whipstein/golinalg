package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dggbal balances a pair of general real matrices (A,B).  This
// involves, first, permuting A and B by similarity transformations to
// isolate eigenvalues in the first 1 to ILO$-$1 and last IHI+1 to N
// elements on the diagonal; and second, applying a diagonal similarity
// transformation to rows and columns ILO to IHI to make the rows
// and columns as close in norm as possible. Both steps are optional.
//
// Balancing may reduce the 1-norm of the matrices, and improve the
// accuracy of the computed eigenvalues and/or eigenvectors in the
// generalized eigenvalue problem A*x = lambda*B*x.
func Dggbal(job byte, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb, ilo, ihi *int, lscale, rscale, work *mat.Vector, info *int) {
	var alpha, basl, beta, cab, cmax, coef, coef2, coef5, cor, ew, ewc, gamma, half, one, pgamma, rab, sclfac, sfmax, sfmin, sum, t, ta, tb, tc, three, zero float64
	var i, icab, iflow, ip1, ir, irab, it, j, jc, jp1, k, kount, l, lcab, lm1, lrab, lsfmax, lsfmin, m, nr, nrp2 int

	zero = 0.0
	half = 0.5
	one = 1.0
	three = 3.0
	sclfac = 1.0e+1

	//     Test the input parameters
	(*info) = 0
	if job != 'N' && job != 'P' && job != 'S' && job != 'B' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGBAL"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		(*ilo) = 1
		(*ihi) = (*n)
		return
	}

	if (*n) == 1 {
		(*ilo) = 1
		(*ihi) = (*n)
		lscale.Set(0, one)
		rscale.Set(0, one)
		return
	}

	if job == 'N' {
		(*ilo) = 1
		(*ihi) = (*n)
		for i = 1; i <= (*n); i++ {
			lscale.Set(i-1, one)
			rscale.Set(i-1, one)
		}
		return
	}

	k = 1
	l = (*n)
	if job == 'S' {
		goto label190
	}

	goto label30

	//     Permute the matrices A and B to isolate the eigenvalues.
	//
	//     Find row with one nonzero in columns 1 through L
label20:
	;
	l = lm1
	if l != 1 {
		goto label30
	}

	rscale.Set(0, one)
	lscale.Set(0, one)
	goto label190

label30:
	;
	lm1 = l - 1
	for i = l; i >= 1; i-- {
		for j = 1; j <= lm1; j++ {
			jp1 = j + 1
			if a.Get(i-1, j-1) != zero || b.Get(i-1, j-1) != zero {
				goto label50
			}
		}
		j = l
		goto label70

	label50:
		;
		for j = jp1; j <= l; j++ {
			if a.Get(i-1, j-1) != zero || b.Get(i-1, j-1) != zero {
				goto label80
			}
		}
		j = jp1 - 1

	label70:
		;
		m = l
		iflow = 1
		goto label160
	label80:
	}
	goto label100

	//     Find column with one nonzero in rows K through N
label90:
	;
	k = k + 1

label100:
	;
	for j = k; j <= l; j++ {
		for i = k; i <= lm1; i++ {
			ip1 = i + 1
			if a.Get(i-1, j-1) != zero || b.Get(i-1, j-1) != zero {
				goto label120
			}
		}
		i = l
		goto label140
	label120:
		;
		for i = ip1; i <= l; i++ {
			if a.Get(i-1, j-1) != zero || b.Get(i-1, j-1) != zero {
				goto label150
			}
		}
		i = ip1 - 1
	label140:
		;
		m = k
		iflow = 2
		goto label160
	label150:
	}
	goto label190

	//     Permute rows M and I
label160:
	;
	lscale.Set(m-1, float64(i))
	if i == m {
		goto label170
	}
	goblas.Dswap(toPtr((*n)-k+1), a.Vector(i-1, k-1), lda, a.Vector(m-1, k-1), lda)
	goblas.Dswap(toPtr((*n)-k+1), b.Vector(i-1, k-1), ldb, b.Vector(m-1, k-1), ldb)

	//     Permute columns M and J
label170:
	;
	rscale.Set(m-1, float64(j))
	if j == m {
		goto label180
	}
	goblas.Dswap(&l, a.Vector(0, j-1), func() *int { y := 1; return &y }(), a.Vector(0, m-1), func() *int { y := 1; return &y }())
	goblas.Dswap(&l, b.Vector(0, j-1), func() *int { y := 1; return &y }(), b.Vector(0, m-1), func() *int { y := 1; return &y }())

label180:
	;
	switch iflow {
	case 1:
		goto label20
	case 2:
		goto label90
	}

label190:
	;
	(*ilo) = k
	(*ihi) = l

	if job == 'P' {
		for i = (*ilo); i <= (*ihi); i++ {
			lscale.Set(i-1, one)
			rscale.Set(i-1, one)
		}
		return
	}

	if (*ilo) == (*ihi) {
		return
	}

	//     Balance the submatrix in rows ILO to IHI.
	nr = (*ihi) - (*ilo) + 1
	for i = (*ilo); i <= (*ihi); i++ {
		rscale.Set(i-1, zero)
		lscale.Set(i-1, zero)

		work.Set(i-1, zero)
		work.Set(i+(*n)-1, zero)
		work.Set(i+2*(*n)-1, zero)
		work.Set(i+3*(*n)-1, zero)
		work.Set(i+4*(*n)-1, zero)
		work.Set(i+5*(*n)-1, zero)
	}

	//     Compute right side vector in resulting linear equations
	basl = math.Log10(sclfac)
	for i = (*ilo); i <= (*ihi); i++ {
		for j = (*ilo); j <= (*ihi); j++ {
			tb = b.Get(i-1, j-1)
			ta = a.Get(i-1, j-1)
			if ta == zero {
				goto label210
			}
			ta = math.Log10(math.Abs(ta)) / basl
		label210:
			;
			if tb == zero {
				goto label220
			}
			tb = math.Log10(math.Abs(tb)) / basl
		label220:
			;
			work.Set(i+4*(*n)-1, work.Get(i+4*(*n)-1)-ta-tb)
			work.Set(j+5*(*n)-1, work.Get(j+5*(*n)-1)-ta-tb)
		}
	}

	coef = one / float64(2*nr)
	coef2 = coef * coef
	coef5 = half * coef2
	nrp2 = nr + 2
	beta = zero
	it = 1

	//     Start generalized conjugate gradient iteration
label250:
	;

	gamma = goblas.Ddot(&nr, work.Off((*ilo)+4*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+4*(*n)-1), func() *int { y := 1; return &y }()) + goblas.Ddot(&nr, work.Off((*ilo)+5*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+5*(*n)-1), func() *int { y := 1; return &y }())

	ew = zero
	ewc = zero
	for i = (*ilo); i <= (*ihi); i++ {
		ew = ew + work.Get(i+4*(*n)-1)
		ewc = ewc + work.Get(i+5*(*n)-1)
	}

	gamma = coef*gamma - coef2*(math.Pow(ew, 2)+math.Pow(ewc, 2)) - coef5*math.Pow(ew-ewc, 2)
	if gamma == zero {
		goto label350
	}
	if it != 1 {
		beta = gamma / pgamma
	}
	t = coef5 * (ewc - three*ew)
	tc = coef5 * (ew - three*ewc)

	goblas.Dscal(&nr, &beta, work.Off((*ilo)-1), func() *int { y := 1; return &y }())
	goblas.Dscal(&nr, &beta, work.Off((*ilo)+(*n)-1), func() *int { y := 1; return &y }())

	goblas.Daxpy(&nr, &coef, work.Off((*ilo)+4*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+(*n)-1), func() *int { y := 1; return &y }())
	goblas.Daxpy(&nr, &coef, work.Off((*ilo)+5*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)-1), func() *int { y := 1; return &y }())

	for i = (*ilo); i <= (*ihi); i++ {
		work.Set(i-1, work.Get(i-1)+tc)
		work.Set(i+(*n)-1, work.Get(i+(*n)-1)+t)
	}

	//     Apply matrix to vector
	for i = (*ilo); i <= (*ihi); i++ {
		kount = 0
		sum = zero
		for j = (*ilo); j <= (*ihi); j++ {
			if a.Get(i-1, j-1) == zero {
				goto label280
			}
			kount = kount + 1
			sum = sum + work.Get(j-1)
		label280:
			;
			if b.Get(i-1, j-1) == zero {
				goto label290
			}
			kount = kount + 1
			sum = sum + work.Get(j-1)
		label290:
		}
		work.Set(i+2*(*n)-1, float64(kount)*work.Get(i+(*n)-1)+sum)
	}

	for j = (*ilo); j <= (*ihi); j++ {
		kount = 0
		sum = zero
		for i = (*ilo); i <= (*ihi); i++ {
			if a.Get(i-1, j-1) == zero {
				goto label310
			}
			kount = kount + 1
			sum = sum + work.Get(i+(*n)-1)
		label310:
			;
			if b.Get(i-1, j-1) == zero {
				goto label320
			}
			kount = kount + 1
			sum = sum + work.Get(i+(*n)-1)
		label320:
		}
		work.Set(j+3*(*n)-1, float64(kount)*work.Get(j-1)+sum)

	}

	sum = goblas.Ddot(&nr, work.Off((*ilo)+(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+2*(*n)-1), func() *int { y := 1; return &y }()) + goblas.Ddot(&nr, work.Off((*ilo)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+3*(*n)-1), func() *int { y := 1; return &y }())
	alpha = gamma / sum

	//     Determine correction to current iteration
	cmax = zero
	for i = (*ilo); i <= (*ihi); i++ {
		cor = alpha * work.Get(i+(*n)-1)
		if math.Abs(cor) > cmax {
			cmax = math.Abs(cor)
		}
		lscale.Set(i-1, lscale.Get(i-1)+cor)
		cor = alpha * work.Get(i-1)
		if math.Abs(cor) > cmax {
			cmax = math.Abs(cor)
		}
		rscale.Set(i-1, rscale.Get(i-1)+cor)
	}
	if cmax < half {
		goto label350
	}

	goblas.Daxpy(&nr, toPtrf64(-alpha), work.Off((*ilo)+2*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+4*(*n)-1), func() *int { y := 1; return &y }())
	goblas.Daxpy(&nr, toPtrf64(-alpha), work.Off((*ilo)+3*(*n)-1), func() *int { y := 1; return &y }(), work.Off((*ilo)+5*(*n)-1), func() *int { y := 1; return &y }())

	pgamma = gamma
	it = it + 1
	if it <= nrp2 {
		goto label250
	}

	//     End generalized conjugate gradient iteration
label350:
	;
	sfmin = Dlamch(SafeMinimum)
	sfmax = one / sfmin
	lsfmin = int(math.Log10(sfmin)/basl + one)
	lsfmax = int(math.Log10(sfmax) / basl)
	for i = (*ilo); i <= (*ihi); i++ {
		irab = goblas.Idamax(toPtr((*n)-(*ilo)+1), a.Vector(i-1, (*ilo)-1), lda)
		rab = math.Abs(a.Get(i-1, irab+(*ilo)-1-1))
		irab = goblas.Idamax(toPtr((*n)-(*ilo)+1), b.Vector(i-1, (*ilo)-1), ldb)
		rab = maxf64(rab, math.Abs(b.Get(i-1, irab+(*ilo)-1-1)))
		lrab = int(math.Log10(rab+sfmin)/basl + one)
		ir = int(lscale.Get(i-1) + signf64(half, lscale.Get(i-1)))
		ir = minint(maxint(ir, lsfmin), lsfmax, lsfmax-lrab)
		lscale.Set(i-1, math.Pow(sclfac, float64(ir)))
		icab = goblas.Idamax(ihi, a.Vector(0, i-1), func() *int { y := 1; return &y }())
		cab = math.Abs(a.Get(icab-1, i-1))
		icab = goblas.Idamax(ihi, b.Vector(0, i-1), func() *int { y := 1; return &y }())
		cab = maxf64(cab, math.Abs(b.Get(icab-1, i-1)))
		lcab = int(math.Log10(cab+sfmin)/basl + one)
		jc = int(rscale.Get(i-1) + signf64(half, rscale.Get(i-1)))
		jc = minint(maxint(jc, lsfmin), lsfmax, lsfmax-lcab)
		rscale.Set(i-1, math.Pow(sclfac, float64(jc)))
	}

	//     Row scaling of matrices A and B
	for i = (*ilo); i <= (*ihi); i++ {
		goblas.Dscal(toPtr((*n)-(*ilo)+1), lscale.GetPtr(i-1), a.Vector(i-1, (*ilo)-1), lda)
		goblas.Dscal(toPtr((*n)-(*ilo)+1), lscale.GetPtr(i-1), b.Vector(i-1, (*ilo)-1), ldb)
	}

	//     Column scaling of matrices A and B
	for j = (*ilo); j <= (*ihi); j++ {
		goblas.Dscal(ihi, rscale.GetPtr(j-1), a.Vector(0, j-1), func() *int { y := 1; return &y }())
		goblas.Dscal(ihi, rscale.GetPtr(j-1), b.Vector(0, j-1), func() *int { y := 1; return &y }())
	}
}
