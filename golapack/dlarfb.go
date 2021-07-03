package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlarfb applies a real block reflector H or its transpose H**T to a
// real m by n matrix C, from either the left or the right.
func Dlarfb(side, trans, direct, storev byte, m, n, k *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, c *mat.Matrix, ldc *int, work *mat.Matrix, ldwork *int) {
	var transt byte
	var one float64
	var i, j int
	var err error
	_ = err

	one = 1.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	if trans == 'N' {
		transt = 'T'
	} else {
		transt = 'N'
	}

	if storev == 'C' {
		if direct == 'F' {
			//           Let  V =  ( V1 )    (first K rows)
			//                     ( V2 )
			//           where  V1  is unit lower triangular.
			if side == 'L' {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
				//
				//              W := C1**T
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*n, c.Vector(j-1, 0), *ldc, work.Vector(0, j-1), 1)
				}

				//              W := W * V1
				err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, *n, *k, one, v, *ldv, work, *ldwork)
				if (*m) > (*k) {
					//                 W := W + C2**T * V2
					err = goblas.Dgemm(Trans, NoTrans, *n, *k, (*m)-(*k), one, c.Off((*k)+1-1, 0), *ldc, v.Off((*k)+1-1, 0), *ldv, one, work, *ldwork)
				}

				//              W := W * T**T  or  W * T
				err = goblas.Dtrmm(Right, Upper, mat.TransByte(transt), NonUnit, *n, *k, one, t, *ldt, work, *ldwork)

				//              C := C - V * W**T
				if (*m) > (*k) {
					//                 C2 := C2 - V2 * W**T
					err = goblas.Dgemm(NoTrans, Trans, (*m)-(*k), *n, *k, -one, v.Off((*k)+1-1, 0), *ldv, work, *ldwork, one, c.Off((*k)+1-1, 0), *ldc)
				}

				//              W := W * V1**T
				err = goblas.Dtrmm(Right, Lower, Trans, Unit, *n, *k, one, v, *ldv, work, *ldwork)

				//              C1 := C1 - W**T
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*m, c.Vector(0, j-1), 1, work.Vector(0, j-1), 1)
				}

				//              W := W * V1
				err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, *m, *k, one, v, *ldv, work, *ldwork)
				if (*n) > (*k) {
					//                 W := W + C2 * V2
					err = goblas.Dgemm(NoTrans, NoTrans, *m, *k, (*n)-(*k), one, c.Off(0, (*k)+1-1), *ldc, v.Off((*k)+1-1, 0), *ldv, one, work, *ldwork)
				}

				//              W := W * T  or  W * T**T
				err = goblas.Dtrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

				//              C := C - W * V**T
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2**T
					err = goblas.Dgemm(NoTrans, Trans, *m, (*n)-(*k), *k, -one, work, *ldwork, v.Off((*k)+1-1, 0), *ldv, one, c.Off(0, (*k)+1-1), *ldc)
				}

				//              W := W * V1**T
				err = goblas.Dtrmm(Right, Lower, Trans, Unit, *m, *k, one, v, *ldv, work, *ldwork)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}
			}

		} else {
			//           Let  V =  ( V1 )
			//                     ( V2 )    (last K rows)
			//           where  V2  is unit upper triangular.
			if side == 'L' {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
				//
				//              W := C2**T
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*n, c.Vector((*m)-(*k)+j-1, 0), *ldc, work.Vector(0, j-1), 1)
				}

				//              W := W * V2
				err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, *n, *k, one, v.Off((*m)-(*k)+1-1, 0), *ldv, work, *ldwork)
				if (*m) > (*k) {
					//                 W := W + C1**T * V1
					err = goblas.Dgemm(Trans, NoTrans, *n, *k, (*m)-(*k), one, c, *ldc, v, *ldv, one, work, *ldwork)
				}

				//              W := W * T**T  or  W * T
				err = goblas.Dtrmm(Right, Lower, mat.TransByte(transt), NonUnit, *n, *k, one, t, *ldt, work, *ldwork)

				//              C := C - V * W**T
				if (*m) > (*k) {
					//                 C1 := C1 - V1 * W**T
					err = goblas.Dgemm(NoTrans, Trans, (*m)-(*k), *n, *k, -one, v, *ldv, work, *ldwork, one, c, *ldc)
				}

				//              W := W * V2**T
				err = goblas.Dtrmm(Right, Upper, Trans, Unit, *n, *k, one, v.Off((*m)-(*k)+1-1, 0), *ldv, work, *ldwork)

				//              C2 := C2 - W**T
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set((*m)-(*k)+j-1, i-1, c.Get((*m)-(*k)+j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*m, c.Vector(0, (*n)-(*k)+j-1), 1, work.Vector(0, j-1), 1)
				}

				//              W := W * V2
				err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, *m, *k, one, v.Off((*n)-(*k)+1-1, 0), *ldv, work, *ldwork)
				if (*n) > (*k) {
					//                 W := W + C1 * V1
					err = goblas.Dgemm(NoTrans, NoTrans, *m, *k, (*n)-(*k), one, c, *ldc, v, *ldv, one, work, *ldwork)
				}

				//              W := W * T  or  W * T**T
				err = goblas.Dtrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

				//              C := C - W * V**T
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1**T
					err = goblas.Dgemm(NoTrans, Trans, *m, (*n)-(*k), *k, -one, work, *ldwork, v, *ldv, one, c, *ldc)
				}

				//              W := W * V2**T
				err = goblas.Dtrmm(Right, Upper, Trans, Unit, *m, *k, one, v.Off((*n)-(*k)+1-1, 0), *ldv, work, *ldwork)

				//              C2 := C2 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, (*n)-(*k)+j-1, c.Get(i-1, (*n)-(*k)+j-1)-work.Get(i-1, j-1))
					}
				}
			}
		}

	} else if storev == 'R' {

		if direct == 'F' {
			//           Let  V =  ( V1  V2 )    (V1: first K columns)
			//           where  V1  is unit upper triangular.
			if side == 'L' {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
				//
				//              W := C1**T
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*n, c.Vector(j-1, 0), *ldc, work.Vector(0, j-1), 1)
				}

				//              W := W * V1**T
				err = goblas.Dtrmm(Right, Upper, Trans, Unit, *n, *k, one, v, *ldv, work, *ldwork)
				if (*m) > (*k) {
					//                 W := W + C2**T * V2**T
					err = goblas.Dgemm(Trans, Trans, *n, *k, (*m)-(*k), one, c.Off((*k)+1-1, 0), *ldc, v.Off(0, (*k)+1-1), *ldv, one, work, *ldwork)
				}

				//              W := W * T**T  or  W * T
				err = goblas.Dtrmm(Right, Upper, mat.TransByte(transt), NonUnit, *n, *k, one, t, *ldt, work, *ldwork)

				//              C := C - V**T * W**T
				if (*m) > (*k) {
					//                 C2 := C2 - V2**T * W**T
					err = goblas.Dgemm(Trans, Trans, (*m)-(*k), *n, *k, -one, v.Off(0, (*k)+1-1), *ldv, work, *ldwork, one, c.Off((*k)+1-1, 0), *ldc)
				}

				//              W := W * V1
				err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, *n, *k, one, v, *ldv, work, *ldwork)

				//              C1 := C1 - W**T
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set(j-1, i-1, c.Get(j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
				//
				//              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
				//
				//              W := C1
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*m, c.Vector(0, j-1), 1, work.Vector(0, j-1), 1)
				}

				//              W := W * V1**T
				err = goblas.Dtrmm(Right, Upper, Trans, Unit, *m, *k, one, v, *ldv, work, *ldwork)
				if (*n) > (*k) {
					//                 W := W + C2 * V2**T
					err = goblas.Dgemm(NoTrans, Trans, *m, *k, (*n)-(*k), one, c.Off(0, (*k)+1-1), *ldc, v.Off(0, (*k)+1-1), *ldv, one, work, *ldwork)
				}

				//              W := W * T  or  W * T**T
				err = goblas.Dtrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C2 := C2 - W * V2
					err = goblas.Dgemm(NoTrans, NoTrans, *m, (*n)-(*k), *k, -one, work, *ldwork, v.Off(0, (*k)+1-1), *ldv, one, c.Off(0, (*k)+1-1), *ldc)
				}

				//              W := W * V1
				err = goblas.Dtrmm(Right, Upper, NoTrans, Unit, *m, *k, one, v, *ldv, work, *ldwork)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, j-1, c.Get(i-1, j-1)-work.Get(i-1, j-1))
					}
				}

			}

		} else {
			//           Let  V =  ( V1  V2 )    (V2: last K columns)
			//           where  V2  is unit lower triangular.
			if side == 'L' {
				//              Form  H * C  or  H**T * C  where  C = ( C1 )
				//                                                    ( C2 )
				//
				//              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
				//
				//              W := C2**T
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*n, c.Vector((*m)-(*k)+j-1, 0), *ldc, work.Vector(0, j-1), 1)
				}

				//              W := W * V2**T
				err = goblas.Dtrmm(Right, Lower, Trans, Unit, *n, *k, one, v.Off(0, (*m)-(*k)+1-1), *ldv, work, *ldwork)
				if (*m) > (*k) {
					//                 W := W + C1**T * V1**T
					err = goblas.Dgemm(Trans, Trans, *n, *k, (*m)-(*k), one, c, *ldc, v, *ldv, one, work, *ldwork)
				}

				//              W := W * T**T  or  W * T
				err = goblas.Dtrmm(Right, Lower, mat.TransByte(transt), NonUnit, *n, *k, one, t, *ldt, work, *ldwork)

				//              C := C - V**T * W**T
				if (*m) > (*k) {
					//                 C1 := C1 - V1**T * W**T
					err = goblas.Dgemm(Trans, Trans, (*m)-(*k), *n, *k, -one, v, *ldv, work, *ldwork, one, c, *ldc)
				}

				//              W := W * V2
				err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, *n, *k, one, v.Off(0, (*m)-(*k)+1-1), *ldv, work, *ldwork)

				//              C2 := C2 - W**T
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*n); i++ {
						c.Set((*m)-(*k)+j-1, i-1, c.Get((*m)-(*k)+j-1, i-1)-work.Get(i-1, j-1))
					}
				}

			} else if side == 'R' {
				//              Form  C * H  or  C * H'  where  C = ( C1  C2 )
				//
				//              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
				//
				//              W := C2
				for j = 1; j <= (*k); j++ {
					goblas.Dcopy(*m, c.Vector(0, (*n)-(*k)+j-1), 1, work.Vector(0, j-1), 1)
				}

				//              W := W * V2**T
				err = goblas.Dtrmm(Right, Lower, Trans, Unit, *m, *k, one, v.Off(0, (*n)-(*k)+1-1), *ldv, work, *ldwork)
				if (*n) > (*k) {
					//                 W := W + C1 * V1**T
					err = goblas.Dgemm(NoTrans, Trans, *m, *k, (*n)-(*k), one, c, *ldc, v, *ldv, one, work, *ldwork)
				}

				//              W := W * T  or  W * T**T
				err = goblas.Dtrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

				//              C := C - W * V
				if (*n) > (*k) {
					//                 C1 := C1 - W * V1
					err = goblas.Dgemm(NoTrans, NoTrans, *m, (*n)-(*k), *k, -one, work, *ldwork, v, *ldv, one, c, *ldc)
				}

				//              W := W * V2
				err = goblas.Dtrmm(Right, Lower, NoTrans, Unit, *m, *k, one, v.Off(0, (*n)-(*k)+1-1), *ldv, work, *ldwork)

				//              C1 := C1 - W
				for j = 1; j <= (*k); j++ {
					for i = 1; i <= (*m); i++ {
						c.Set(i-1, (*n)-(*k)+j-1, c.Get(i-1, (*n)-(*k)+j-1)-work.Get(i-1, j-1))
					}
				}

			}

		}
	}
}
