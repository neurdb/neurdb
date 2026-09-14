import numpy as np


class Expr:
    def __init__(self, expr, list_kind=0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0

    def isCol(
        self,
    ):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def getValue(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"].replace("'", "''") + "'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise "unknown Value in Expr"
        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]["typeName"]["TypeName"]["names"]) == 1:
                return (
                    value_expr["TypeCast"]["typeName"]["TypeName"]["names"][0][
                        "String"
                    ]["str"]
                    + " '"
                    + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"]["str"]
                    + "'"
                )
            else:
                try:
                    # Case: WHERE (...) INTERVAL '1' year
                    if (
                        value_expr["TypeCast"]["typeName"]["TypeName"]["typmods"][0][
                            "A_Const"
                        ]["val"]["Integer"]["ival"]
                        == 2
                    ):
                        return (
                            value_expr["TypeCast"]["typeName"]["TypeName"]["names"][1][
                                "String"
                            ]["str"]
                            + " '"
                            + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"][
                                "str"
                            ]
                            + "' month"
                        )
                    else:
                        return (
                            value_expr["TypeCast"]["typeName"]["TypeName"]["names"][1][
                                "String"
                            ]["str"]
                            + " '"
                            + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"][
                                "str"
                            ]
                            + "' year"
                        )
                except KeyError:
                    # Case: WHERE (...) '1 year'::interval
                    # --> Breaks the above condition, as the 'typmods' key is not present
                    #
                    # 'interval'
                    casted_type = value_expr["TypeCast"]["typeName"]["TypeName"][
                        "names"
                    ][1]["String"]["str"]
                    # value: '1 year'
                    string_value = value_expr["TypeCast"]["arg"]["A_Const"]["val"][
                        "String"
                    ]["str"]
                    return f"'{string_value}'::{casted_type}"

        else:
            print(value_expr.keys())
            raise "unknown Value in Expr"

    def getAliasName(
        self,
    ):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(
        self,
    ):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(
        self,
    ):
        if self.isCol():
            return self.getAliasName() + "." + self.getColumnName()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "(" + ",\n".join([self.getValue(x) for x in self.expr]) + ")"
            elif self.list_kind == 10:
                return " AND ".join([self.getValue(x) for x in self.expr])
            else:
                raise "list kind error"

        else:
            raise "No Known type of Expr"


class SpecialSTACKPlusExpr(Expr):
    def isCol(
        self,
    ):
        return True

    def getValue(self, value_expr=None):
        alias = self.getAliasName()
        column = self.getColumnName()

        interval_size = self.expr["A_Expr"]["rexpr"]["TypeCast"]["arg"]["A_Const"][
            "val"
        ]["String"]["str"]
        return f"{alias}.{column} + '{interval_size}'::interval"

    def getAliasName(
        self,
    ):
        return self.expr["A_Expr"]["lexpr"]["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(
        self,
    ):
        return self.expr["A_Expr"]["lexpr"]["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(
        self,
    ):
        return self.getValue()


class TargetTable:
    def __init__(self, target):
        """
        {'location': 7, 'name': 'alternative_name', 'val': {'FuncCall': {'funcname': [{'String': {'str': 'min'}}], 'args': [{'ColumnRef': {'fields': [{'String': {'str': 'an'}}, {'String': {'str': 'name'}}], 'location': 11}}], 'location': 7}}}
        """
        self.target = target

    #         print(self.target)

    def getValue(
        self,
    ):
        columnRef = self.target["val"]["FuncCall"]["args"][0]["ColumnRef"]["fields"]
        return columnRef[0]["String"]["str"] + "." + columnRef[1]["String"]["str"]

    def __str__(
        self,
    ):
        try:
            return (
                self.target["val"]["FuncCall"]["funcname"][0]["String"]["str"]
                + "("
                + self.getValue()
                + ")"
                + " AS "
                + self.target["name"]
            )
        except:
            if "FuncCall" in self.target["val"]:
                return "count(*)"
            else:
                return "*"


class FromTable:
    def __init__(self, from_table):
        """
        {'alias': {'Alias': {'aliasname': 'an'}}, 'location': 168, 'inhOpt': 2, 'relpersistence': 'p', 'relname': 'aka_name'}
        """
        self.from_table = from_table
        if not "alias" in self.from_table:
            self.from_table["alias"] = {"Alias": {"aliasname": from_table["relname"]}}

    def getFullName(
        self,
    ):
        return self.from_table["relname"]

    def getAliasName(
        self,
    ):
        return self.from_table["alias"]["Alias"]["aliasname"]

    def __str__(
        self,
    ):
        try:
            return self.getFullName() + " AS " + self.getAliasName()
        except:
            print(self.from_table)
            raise


class Comparison:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.column = str(self.lexpr)
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"], self.kind)
            elif (
                comparison["A_Expr"]["rexpr"]["A_Expr"]["name"][0]["String"]["str"]
                == "+"
            ):
                self.rexpr = SpecialSTACKPlusExpr(
                    comparison["A_Expr"]["rexpr"], self.kind
                )

            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.column = str(self.lexpr)
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x) for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.aliasname_list.append(comp.lexpr.getAliasName())
                    self.lexpr = comp.lexpr
                    self.column = str(self.lexpr)
                    self.column_list.append(comp.lexpr.getColumnName())
                    break
            self.comp_kind = 2

    def isCol(
        self,
    ):
        return False

    def __str__(
        self,
    ):

        if self.comp_kind == 0:
            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 8:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "~~*":
                    Op = "ilike"
                else:
                    raise
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json

                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise "Operation ERROR"
            return str(self.lexpr) + " " + Op + " " + str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr) + " IS NOT NULL"
            else:
                return str(self.lexpr) + " IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( " + str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res


class Table:
    def __init__(self, table_tree):
        self.name = table_tree["relation"]["RangeVar"]["relname"]
        self.column2idx = {}
        self.idx2column = {}
        self.column2type = {}
        for idx, columndef in enumerate(table_tree["tableElts"]):
            self.column2idx[columndef["ColumnDef"]["colname"]] = idx
            self.column2type[columndef["ColumnDef"]["colname"]] = columndef[
                "ColumnDef"
            ]["typeName"]["TypeName"]["names"][-1]["String"]["str"]
            # print(columndef["ColumnDef"]["typeName"]['TypeName']['names'],self.column2type[columndef["ColumnDef"]["colname"]],self.column2type[columndef["ColumnDef"]["colname"]] in ['int4','text','varchar'])
            assert self.column2type[columndef["ColumnDef"]["colname"]] in [
                "int4",
                "text",
                "varchar",
            ]
            if self.column2type[columndef["ColumnDef"]["colname"]] == "int4":
                self.column2type[columndef["ColumnDef"]["colname"]] = "int"
            else:
                self.column2type[columndef["ColumnDef"]["colname"]] = "str"
            self.idx2column[idx] = columndef["ColumnDef"]["colname"]

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))


class DB:
    def __init__(self, schema, TREE_NUM_IN_NET=40):
        from psqlparse import parse_dict

        parse_tree = parse_dict(schema)

        self.tables = []
        self.name2idx = {}
        self.table_names = []
        self.name2table = {}
        self.size = 0
        self.TREE_NUM_IN_NET = TREE_NUM_IN_NET
        for idx, table_tree in enumerate(parse_tree):
            self.tables.append(Table(table_tree["CreateStmt"]))
            self.table_names.append(self.tables[-1].name)
            self.name2idx[self.tables[-1].name] = idx
            self.name2table[self.tables[-1].name] = self.tables[-1]

        self.columns_total = 0

        for table in self.tables:
            self.columns_total += len(table.idx2column)

        self.size = len(self.table_names)

    def is_str(self, table, column):
        table = self.name2table[table]
        return table.column2type[column] == "str"

    def __len__(
        self,
    ):
        if self.size == 0:
            self.size = len(self.table_names)
        return self.size

    def oneHotAll(
        self,
    ):
        return np.zeros((1, self.size))

    def network_size(
        self,
    ):
        return self.TREE_NUM_IN_NET * self.size
